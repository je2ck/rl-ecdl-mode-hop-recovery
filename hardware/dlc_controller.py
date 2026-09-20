import math
import os
import random
from toptica.lasersdk.client import NetworkConnection
from toptica.lasersdk.dlcpro.v3_2_0 import DLCpro


class DLC:
    def __init__(self, connection_ip: str, is_dummy=False):
        self.connection_ip = connection_ip
        self.is_dummy = is_dummy

        if not self.is_dummy:
            self.connection = DLCpro(NetworkConnection(self.connection_ip))
        else:
            self.connection = None

    def __enter__(self):
        if not self.is_dummy:
            self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_dummy:
            self.connection.__exit__(exc_type, exc_val, exc_tb)

    def get_uptime(self) -> str:
        if self.is_dummy:
            return f"{random.randint(0, 9999)}s"
        return self.connection.uptime_txt.get()

    def get_current(self) -> float:
        if self.is_dummy:
            return round(random.uniform(0.0, 200.0), 2)
        return self.connection.laser1.dl.cc.current_set.get()

    def get_act_current(self) -> float:
        if self.is_dummy:
            return round(random.uniform(0.0, 200.0), 2)
        return self.connection.laser1.dl.cc.current_act.get()

    def set_current(self, new_current: float):
        if self.is_dummy:
            return

        self.connection.laser1.dl.cc.current_set.set(new_current)

        set_value = self.get_current()
        if not math.isclose(set_value, new_current, abs_tol=1e-6):
            raise ValueError(
                f"Failed to set laser current to {new_current}. Current value is {set_value}."
            )

    def get_voltage(self) -> float:
        if self.is_dummy:
            return round(random.uniform(-5.0, 5.0), 2)
        return self.connection.laser1.dl.pc.voltage_set.get()

    def get_act_voltage(self) -> float:
        if self.is_dummy:
            return round(random.uniform(-5.0, 5.0), 2)
        return self.connection.laser1.dl.pc.voltage_act.get()

    def set_voltage(self, new_voltage: float):
        if self.is_dummy:
            return

        self.connection.laser1.dl.pc.voltage_set.set(new_voltage)

        set_value = self.get_voltage()
        if not math.isclose(set_value, new_voltage, abs_tol=1e-6):
            raise ValueError(
                f"Failed to set laser voltage to {new_voltage}. Current value is {set_value}."
            )

    def get_temp(self) -> float:
        if self.is_dummy:
            return round(random.uniform(0.0, 200.0), 2)
        return self.connection.laser1.dl.tc.temp_set.get()

    def get_act_temp(self) -> float:
        if self.is_dummy:
            return round(random.uniform(0.0, 200.0), 2)
        return self.connection.laser1.dl.tc.temp_act.get()

    def set_temp(self, new_temp: float):
        if self.is_dummy:
            return

        self.connection.laser1.dl.tc.temp_set.set(new_temp)

        set_value = self.get_temp()
        if not math.isclose(set_value, new_temp, abs_tol=1e-6):
            raise ValueError(
                f"Failed to set laser temperature to {new_temp}. Temperature value is {set_value}."
            )


if __name__ == "__main__":
    dlc_host = os.environ.get("DLC_HOST") or os.environ.get("DLC_IP")
    if not dlc_host:
        raise SystemExit("Set DLC_HOST before running this hardware check.")
    dlc_controller = DLC(dlc_host)
    with dlc_controller as dlc:
        print(dlc.get_current())
        print(dlc.get_act_current())
