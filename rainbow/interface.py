import numpy as np
import math
import cv2
from PIL import Image
from datetime import datetime

from laser_sim import PyDrawer, PyDiodeLaser, Signal
from .constants import (
    CURRENT_LONG_RANGE,
    CURRENT_RANGE,
    FREQUENCY_RANGE,
    IMAGE_SIZE,
    SENSOR_DIM,
)


class LaserInterface:
    def __init__(
        self,
        target_frequency,
        current_range="short",
        random_target=False,
        api_client=None,
        dlc_controller=None,
        osc=None,
        is_test=False,
        plotter=None,
    ):
        self.stage_count = 0
        self.params = {}
        self.episode_frames = 0
        self.max_frames = 2000
        self.random_target = random_target
        self.target_frequency = target_frequency
        self.game_over_flag = False
        self.image = np.zeros(
            (int(FREQUENCY_RANGE), int(FREQUENCY_RANGE), 3), dtype=np.uint8
        )
        self.sensor_data = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.is_simulation = api_client is None or dlc_controller is None or osc is None
        self.is_test = is_test
        self.current_range = (
            CURRENT_RANGE if current_range == "short" else CURRENT_LONG_RANGE
        )
        if not self.is_simulation:
            from hardware.laser import DiodeLaser

            self.laser = DiodeLaser(api_client, dlc_controller, osc, plotter)
        else:
            self.laser = PyDiodeLaser(int(self.current_range))

        self.params["model_current"] = True
        self.sequence_window = 12
        self.penalty_sequence = []
        self.penalty_count = 0
        self.drawer = PyDrawer(int(self.current_range))
        if not is_test:
            self.drawer.with_fix_monitor(True)
        self.truncate = False

    def setInt(self, key: str, value: int):
        self.params[key] = value
        if key == "max_num_frames_per_episode":
            self.max_frames = value

    def setFloat(self, key: str, value: float):
        self.params[key] = value

    def setBool(self, key: str, value: bool):
        self.params[key] = value

    def getMinimalActionSet(self):
        if self.params["model_current"]:
            return [0, 1, 2]
        return [0, 1, 2, 3, 4]

    def act(self, action):
        reward = 0
        done = False
        self.truncate = False
        self.episode_frames += 1

        if action == 0:
            self.laser_state = self.laser.latest()
        elif action == 1:
            if self.laser_state.current + 0.1 >= self.current_upper_bound + 1e-9:
                self.truncate = True
            self.laser_state, amount = self.laser.emit(Signal.CurrentUp)
            self.drawer.update(self.laser_state, Signal.CurrentUp, amount)
        elif action == 2:
            if self.laser_state.current - 0.1 < self.current_lower_bound + 1e-9:
                self.truncate = True
            self.laser_state, amount = self.laser.emit(Signal.CurrentDown)
            self.drawer.update(self.laser_state, Signal.CurrentDown, amount)
        elif action == 3:
            self.laser_state, amount = self.laser.emit(Signal.PZTUp)
            self.drawer.update(self.laser_state, Signal.PZTUp, amount)
        elif action == 4:
            self.laser_state, amount = self.laser.emit(Signal.PZTDown)
            self.drawer.update(self.laser_state, Signal.PZTDown, amount)

        freq_diff, current_range, pzt_range = self._update_sensor_data(
            self.laser_state, action
        )

        freq_diff_int = freq_diff * 10**5
        if self.params["model_current"]:
            if self.laser_state.stable and pzt_range:
                reward = 1.0
                done = True
        else:
            if self.laser_state.stable and pzt_range:
                if freq_diff_int <= 3:
                    reward = 1.0
                    done = True
                elif freq_diff_int <= 10:
                    reward = math.exp(-(freq_diff_int / 100))
            else:
                self.truncate = True

        if current_range:
            if len(self.penalty_sequence) < self.sequence_window:
                self.penalty_sequence.append(self.laser_state.current)
            else:
                if not self.analyze_sequence(self.penalty_sequence):
                    reward = -1
                    self.penalty_count += 1
                else:
                    self.penalty_count = 0
                self.penalty_sequence = []

        current_out_of_range = (
            self.laser_state.current < self.current_lower_bound
            or self.laser_state.current > self.current_upper_bound
        )
        pzt_out_of_range = (
            self.laser_state.voltage < self.voltage_lower_bound
            or self.laser_state.voltage > self.voltage_upper_bound
        )

        self.truncate = (
            self.truncate
            or current_out_of_range
            or pzt_out_of_range
            or self.penalty_count > 2
            or self.episode_frames >= self.max_frames
        )
        if self.truncate:
            reward = -1
        death = done or self.truncate
        if death:
            self.game_over_flag = True
            self.penalty_count = 0
            if not done:
                self.stage_count = 0
            else:
                self.stage_count += 1
        return reward

    def _update_image(self):
        img_arr = self.drawer.draw(self.laser_state)
        img = np.frombuffer(img_arr, dtype=np.uint8).reshape(
            (int(FREQUENCY_RANGE), int(FREQUENCY_RANGE), 3)
        )
        img = img[..., ::-1]
        assert int(FREQUENCY_RANGE) > 512
        image = Image.fromarray(img, mode="RGB")
        resized_image = image.resize((512, 512), Image.NEAREST)
        self.image = np.array(resized_image)

    def _update_sensor_data(self, laser_state, action):
        frequency = laser_state.frequency
        freq_diff = abs(frequency - self.target_frequency)
        if not laser_state.stable:
            freq_diff = 1.0
        freq_diff_int = freq_diff * 10**5
        pzt_range = freq_diff_int < 1000
        current_range = not pzt_range
        self.sensor_data = (
            freq_diff,
            float(laser_state.current) - self.current_lower_bound,
            float(laser_state.voltage) - self.voltage_lower_bound,
            float(pzt_range),
            float(current_range),
            float(laser_state.stable),
            float(action),
        )
        return freq_diff, current_range, pzt_range

    def analyze_sequence(self, seq):
        n = len(seq)
        diff = [seq[i + 1] - seq[i] for i in range(n - 1)]

        if all(d == 1 for d in diff):
            return True
        if all(d == -1 for d in diff):
            return True
        if all(d == 0 for d in diff):
            return False

        for p in range(1, len(diff) + 1):
            pattern = diff[:p]
            if sum(pattern) != 0:
                continue
            valid = True
            for i in range(len(diff)):
                if diff[i] != pattern[i % p]:
                    valid = False
                    break
            if valid:
                return False

        return True

    def game_over(self) -> bool:
        return self.game_over_flag

    def reset_game(self):
        self.episode_frames = 0
        self.game_over_flag = False
        self.truncate = False
        self.penalty_sequence = []
        self.penalty_count = 0

        if self.random_target:
            target_state = self.drawer.select_random_target(self.stage_count)
            self.target_frequency = target_state.frequency
        else:
            target_state = self.drawer.get_target_from(self.target_frequency)

        self.current_lower_bound = (
            120.0
            if self.current_range == CURRENT_LONG_RANGE
            else target_state.current - self.current_range / 20.0
        )
        self.current_upper_bound = (
            139.0
            if self.current_range == CURRENT_LONG_RANGE
            else target_state.current + self.current_range / 20.0
        )
        self.voltage_lower_bound = target_state.voltage - 1.0
        self.voltage_upper_bound = target_state.voltage + 1.0
        self.drawer.with_target_frequency(self.target_frequency)
        if self.current_range == CURRENT_RANGE:
            self.drawer.with_relative_frame(target_state.current)

        if not self.is_simulation and self.is_test:
            self.laser_state = self.laser.latest()
        else:
            self.laser_state = self.laser.rand_state(
                self.params["random_seed"],
                self.current_lower_bound,
                self.current_upper_bound,
                self.target_frequency,
                self.params["is_validate"],
            )
            cnt = 0
            while round(self.laser_state.frequency, 4) == round(
                self.target_frequency, 4
            ):
                self.laser_state = self.laser.rand_state(
                    self.params["random_seed"],
                    self.current_lower_bound,
                    self.current_upper_bound,
                    self.target_frequency,
                    self.params["is_validate"],
                )
                cnt += 1
                if cnt > 100:
                    print("failed to generate random state out of target state")
                    break
        self.laser.reset(self.laser_state)
        self.drawer.reset(self.laser_state)
        self._update_sensor_data(self.laser_state, action=0)

    def getScreenGrayscale(self):
        self._update_image()
        resized = cv2.resize(
            self.image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def getScreenRGB(self):
        self._update_image()
        return self.image

    def getSensorData(self):
        return self.sensor_data

    def isTargetStable(self):
        laser_state = self.laser.latest()
        frequency = laser_state.frequency
        freq_diff = abs(frequency - self.target_frequency)
        if not laser_state.stable:
            freq_diff = 1.0
        freq_diff_int = freq_diff * 10**5
        pzt_range = freq_diff_int < 1000
        print(
            f"[{datetime.now().strftime('%m%d_%H:%M:%S')}] stable: {laser_state.stable}, frequency: {laser_state.frequency}"
        )
        return laser_state.stable and pzt_range

    def dumpDrawer(self):
        self.drawer.dump()

    def getTruncate(self):
        return self.truncate

    def getNumAction(self):
        return self.episode_frames
