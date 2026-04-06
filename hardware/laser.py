import random
import time
from laser_sim import LaserState, Signal
from .cavity_classifier import WaveformClassifier


def fetch_data_safe(dlc_controller, api_client):
    """Fetch current laser parameters from hardware."""
    current = dlc_controller.get_current()
    voltage = dlc_controller.get_voltage()
    frequency_raw = api_client.get_frequency()
    frequency = (
        float(frequency_raw) / 1e3
        if float(frequency_raw) > 1e6
        else float(frequency_raw)
    )
    return {"current": current, "voltage": voltage, "frequency": frequency}


class DiodeLaser:
    def __init__(self, api_client, dlc_controller, osc, plotter=None):
        self.api_client = api_client  # wavemeter
        self.dlc_controller = dlc_controller  # laser controller
        self.osc = osc  # FP cavity oscilloscope
        self.classifier = WaveformClassifier()
        self.latest_state = None
        self.plotter = plotter

    def emit(self, signal: Signal) -> tuple[LaserState, float]:
        current_act = self.dlc_controller.get_act_current()
        current = self.dlc_controller.get_current()
        voltage_act = round(self.dlc_controller.get_act_voltage(), 1)
        voltage = self.dlc_controller.get_voltage()
        repeat = 20

        if signal == Signal.from_str("CurrentUp"):
            for _ in range(repeat):
                current = self.dlc_controller.get_current()
                current_act = self.dlc_controller.get_act_current()
                self.dlc_controller.set_current(current + 0.1)
                self.check_act_ctrl(current_act)

            for _ in range(repeat - 1):
                current = self.dlc_controller.get_current()
                current_act = self.dlc_controller.get_act_current()
                self.dlc_controller.set_current(current - 0.1)
                self.check_act_ctrl(current_act)

        elif signal == Signal.from_str("CurrentDown"):
            self.dlc_controller.set_current(current - 0.1)
            self.check_act_ctrl(current_act)

        elif signal == Signal.from_str("PZTUp"):
            self.dlc_controller.set_voltage(voltage + 0.1)
            self.check_act_ctrl(voltage_act)

        elif signal == Signal.from_str("PZTDown"):
            self.dlc_controller.set_voltage(voltage - 0.1)
            self.check_act_ctrl(voltage_act)

        state = self.fetch_data()

        move = 0.0
        if signal == Signal.PZTUp or signal == Signal.PZTDown:
            move = state.frequency - self.latest_state.frequency
        self.latest_state = state
        return (state, move)

    def check_act_ctrl(self, current_act):
        new_current_act = self.dlc_controller.get_act_current()
        while abs(current_act - new_current_act) < 0.06:
            new_current_act = self.dlc_controller.get_act_current()

    def fetch_data(self) -> LaserState:
        obs_data = fetch_data_safe(self.dlc_controller, self.api_client)
        while obs_data["frequency"] < 0:
            obs_data = fetch_data_safe(self.dlc_controller, self.api_client)

        time_axis, data_points, voltage = self.osc.get_trim_waveform()
        stable = self.classifier.classify(
            time_axis, data_points, voltage, obs_data["frequency"], self.plotter
        )
        return LaserState(
            frequency=obs_data["frequency"],
            current=obs_data["current"],
            voltage=obs_data["voltage"],
            stable=stable == "Stable",
        )

    def latest(self) -> LaserState:
        state = self.fetch_data()
        self.latest_state = state
        return self.latest_state

    def rand_state(self, seed, lower, upper) -> LaserState:
        if seed != 0:
            random.seed(seed)

        current = random.uniform(lower, upper)
        self._go_to_current(current)
        return self.fetch_data()

    def _go_to_current(self, target_current: float):
        current = self.dlc_controller.get_current()
        if current < target_current:
            while current < target_current:
                self.dlc_controller.set_current(current + 0.1)
                time.sleep(0.3)
                current = self.dlc_controller.get_current()
        elif current > target_current:
            while current > target_current:
                self.dlc_controller.set_current(current - 0.1)
                time.sleep(0.3)
                current = self.dlc_controller.get_current()

    def reset(self, state: LaserState):
        self.osc.close()
        self.osc.connect()
        self._go_to_current(state.current)

    def get_laser_state(self, current: float) -> LaserState:
        self._go_to_current(current)
        return self.fetch_data()
