# -*- coding: utf-8 -*-
import os
from collections import deque
from .interface import LaserInterface, IMAGE_SIZE, SENSOR_DIM
import random
import cv2
import torch
import numpy as np
from datetime import datetime
import time

from hardware.oscilloscope import RigolOscilloscope
from hardware.dlc_controller import DLC
from hardware.wavemeter import WavemeterAPI
from hardware.plotter import PlotManager
from .model import DQN, ImprovedDQN
from .cam import GradCAM


class Env:
    def __init__(self, args):
        self.device = args.device
        self.history_length = args.history_length
        self.is_simulation = args.simulation
        self.image_only = args.image_only
        self.frame_skip_num = args.frame_skip_num
        if args.simulation:
            self.ale = LaserInterface(
                args.target_frequency, args.current_range, args.random_target
            )
        else:
            base_url = os.environ.get("WAVEMETER_URL", "http://localhost")
            DLC_IP = os.environ.get("DLC_IP", "localhost")
            port = int(os.environ.get("WAVEMETER_PORT", "8000"))
            api_client = WavemeterAPI(base_url, port)
            self.dlc_controller = DLC(DLC_IP).__enter__()
            self.oscilloscope = RigolOscilloscope()
            self.oscilloscope.connect()
            plotter = PlotManager()
            self.ale = LaserInterface(
                args.target_frequency,
                args.current_range,
                args.random_target,
                api_client,
                self.dlc_controller,
                self.oscilloscope,
                is_test=True,
                plotter=plotter,
            )
        self.ale.setInt("random_seed", args.seed)
        self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
        self.ale.setFloat("repeat_action_probability", 0)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.setBool("is_validate", args.validation)
        if args.model_current_only:
            self.ale.setBool("model_current", True)
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.life_termination = False
        self.window = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True
        if args.render_with_gradcam:
            if args.use_deep_conv:
                self.dqn = ImprovedDQN(args, action_space=len(self.actions))
            else:
                self.dqn = DQN(args, action_space=len(self.actions))
            ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
            self.dqn.load_state_dict(ckpt)
            self.dqn.to(self.device)
            self.dqn.eval()
            # Hook into last conv layer
            self.gradcam = GradCAM(self.dqn, args.use_deep_conv)
            self.gc_args = args

    def _get_state(self):
        img = cv2.resize(
            self.ale.getScreenGrayscale(),
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.tensor(img, dtype=torch.float32, device=self.device).div_(
            255
        )
        if self.image_only:
            return {"image": image_tensor}

        freq_diff, current, voltage, pzt_range, current_range, is_stable, action = (
            self.ale.getSensorData()
        )
        sensor_tensor = torch.tensor(
            [freq_diff, current, voltage, pzt_range, current_range, is_stable, action],
            dtype=torch.float32,
            device=self.device,
        )
        return {"image": image_tensor, "sensor": sensor_tensor}

    def _reset_buffer(self):
        self.state_buffer = deque([], maxlen=self.history_length)
        default_state = {
            "image": torch.zeros(
                IMAGE_SIZE, IMAGE_SIZE, device=self.device, dtype=torch.float32
            ),
        }
        if not self.image_only:
            default_state["sensor"] = torch.zeros(
                SENSOR_DIM, device=self.device, dtype=torch.float32
            )
        for _ in range(self.window):
            self.state_buffer.append(default_state)

    def change_temperature(self, up: bool):
        curr_temp = self.dlc_controller.get_temp()
        print(
            f"[{datetime.now().strftime('%m%d_%H:%M:%S')}] Before: Temperature now {curr_temp} C"
        )
        target_temp = curr_temp + 0.1 if up else curr_temp - 0.1
        self.dlc_controller.set_temp(target_temp)
        act_temp = self.dlc_controller.get_act_temp()
        while abs(act_temp - target_temp) < 0.02:
            act_temp = self.dlc_controller.get_act_temp()
        print(
            f"[{datetime.now().strftime('%m%d_%H:%M:%S')}] After: Temperature now {target_temp} C"
        )
        time.sleep(10)

    def reset(self):
        if self.life_termination:
            self.life_termination = False
            self.ale.act(0)
        else:
            self._reset_buffer()
            self.ale.reset_game()
            for _ in range(random.randrange(1)):
                self.ale.act(0)
                if self.ale.game_over():
                    self.ale.reset_game()
        observation = self._get_state()
        self.state_buffer.append(observation)

        images = torch.stack([state["image"] for state in self.state_buffer], dim=0)
        if self.image_only:
            return {"image": images}
        sensors = torch.stack([state["sensor"] for state in self.state_buffer], dim=0)
        return {"image": images, "sensor": sensors}

    def _step(self, action):
        reward = self.ale.act(self.actions.get(action))
        state = self._get_state()
        done = self.ale.game_over()

        observation = {
            "image": state["image"],
        }
        if not self.image_only:
            observation["sensor"] = state["sensor"]
        self.state_buffer.append(observation)

        images = torch.stack([s["image"] for s in self.state_buffer], dim=0)
        final_observation = {"image": images}

        if not self.image_only:
            sensors = torch.stack([s["sensor"] for s in self.state_buffer], dim=0)
            final_observation["sensor"] = sensors

        return final_observation, reward, done

    def step(self, action):
        frame_buffer = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        if not self.image_only:
            sensor_buffer = torch.zeros(2, SENSOR_DIM, device=self.device)
        reward = 0
        done = False

        for t in range(self.frame_skip_num):
            reward += self.ale.act(self.actions.get(action))

            if t == self.frame_skip_num - 2:
                state = self._get_state()
                frame_buffer[0] = state["image"]
                if not self.image_only:
                    sensor_buffer[0] = state["sensor"]
            elif t == self.frame_skip_num - 1:
                state = self._get_state()
                frame_buffer[1] = state["image"]
                if not self.image_only:
                    sensor_buffer[1] = state["sensor"]

            done = self.ale.game_over()
            if done:
                break

        img_observation = frame_buffer.max(dim=0)[0]

        if not self.image_only:
            sensor_observation = sensor_buffer.max(dim=0)[0]

        observation = {
            "image": img_observation,
        }

        if not self.image_only:
            observation["sensor"] = sensor_observation

        self.state_buffer.append(observation)

        images = torch.stack([s["image"] for s in self.state_buffer], dim=0)
        final_observation = {"image": images}

        if not self.image_only:
            sensors = torch.stack([s["sensor"] for s in self.state_buffer], dim=0)
            final_observation["sensor"] = sensors

        return final_observation, reward, done

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def is_target_stable(self):
        return self.ale.isTargetStable()

    def dump_drawer(self):
        self.ale.dumpDrawer()

    def action_space(self):
        return len(self.actions)

    def render(self):
        frame_rgb = self.ale.getScreenRGB()

        if hasattr(self, "gradcam"):
            gray_stack = torch.stack([s["image"] for s in self.state_buffer], dim=0)
            x = {"image": gray_stack.unsqueeze(0).to(self.device)}
            cam_map, action_idx = self.gradcam(x, self.gc_args)

            cam_up = cv2.resize(cam_map, (512, 512), interpolation=cv2.INTER_LINEAR)

            heat = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
            heat = heat.astype(np.float32) / 255
            base = frame_rgb.astype(np.float32) / 255
            overlay = heat * 0.5 + base * 0.5
            overlay = np.uint8(255 * overlay)

            cv2.putText(
                overlay,
                f"Action: {action_idx}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("GradCAM Render", overlay)
        else:
            cv2.imshow("screen (RGB)", frame_rgb)

        cv2.waitKey(1)

    def get_truncate(self):
        return self.ale.getTruncate()

    def get_num_action(self):
        return self.ale.getNumAction()

    def close(self):
        if not self.is_simulation:
            self.oscilloscope.close()
            self.dlc_controller.__exit__(None, None, None)
        cv2.destroyAllWindows()
