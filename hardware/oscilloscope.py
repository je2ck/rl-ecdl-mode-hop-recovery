import pyvisa
import json
import numpy as np
import matplotlib.pyplot as plt
import time


class RigolOscilloscope:
    def __init__(self, resource_string="USB", is_dummy=False, channel="CHAN1", timeout=5000):
        self.resource_string = resource_string
        self.channel = channel
        self.timeout = timeout
        self.is_dummy = is_dummy
        self.rm = pyvisa.ResourceManager()
        self.oscilloscope = None

    def connect(self, max_retries=3, delay=2):
        if self.is_dummy:
            return

        for attempt in range(max_retries):
            try:
                if self.resource_string == "USB":
                    resources = self.rm.list_resources()
                    usb_resources = [r for r in resources if "USB" in r]
                    if not usb_resources:
                        raise Exception("No USB oscilloscope found.")
                    self.resource_string = usb_resources[0]
                    print(f"Detected USB oscilloscope: {self.resource_string}")

                self.oscilloscope = self.rm.open_resource(self.resource_string)
                self.oscilloscope.timeout = self.timeout
                self.oscilloscope.chunk_size = 102400
                print("Connected to:", self.oscilloscope.query("*IDN?"))
                return

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                self.oscilloscope = None
                time.sleep(delay)

        print("Failed to connect after multiple attempts.")

    def get_waveform(self, channel=None):
        if not self.oscilloscope:
            print("Oscilloscope is not connected.")
            return None, None

        if channel is None:
            channel = self.channel

        try:
            self.oscilloscope.write(f":WAV:SOUR {channel}")
            self.oscilloscope.write(":WAV:MODE RAW")
            self.oscilloscope.write(":WAV:FORM BYTE")
            self.oscilloscope.write(":WAV:POIN MAX")

            preamble = self.oscilloscope.query(":WAV:PRE?").split(",")
            x_increment = float(preamble[4])
            x_origin = float(preamble[5])
            y_increment = float(preamble[7])
            y_origin = float(preamble[8])
            offset = float(self.oscilloscope.query(":CHAN1:OFFS?"))

            self.oscilloscope.write(":WAV:DATA?")
            raw_data = self.oscilloscope.read_raw()[10:]  # skip SCPI header

            data_points = np.frombuffer(raw_data, dtype=np.uint8)
            voltage = data_points * y_increment + offset
            time_axis = np.arange(len(voltage)) * x_increment + x_origin

            return time_axis, voltage

        except Exception as e:
            print(f"Error retrieving waveform from {channel}:", e)
            return None, None

    def get_trim_waveform(self):
        time1_list = []
        signal1_list = []
        time2_list = []
        signal2_list = []

        for _ in range(5):
            t1, s1 = self.get_waveform(channel="CHAN1")  # transmission/reflection
            t2, s2 = self.get_waveform(channel="CHAN3")  # frequency ramp

            time1_list.append(t1)
            signal1_list.append(s1)
            time2_list.append(t2)
            signal2_list.append(s2)

        time1_array = np.array(time1_list)
        time2_array = np.array(time2_list)
        signal1_array = np.array(signal1_list)
        signal2_array = np.array(signal2_list)

        time1 = np.max(time1_array, axis=0)
        time2 = np.max(time2_array, axis=0)
        signal1 = np.max(signal1_array, axis=0)
        signal2 = np.max(signal2_array, axis=0)

        min_idx = np.argmin(signal2)
        max_idx = np.argmax(signal2)
        if time1 is None or time2 is None:
            return None, None

        half = len(signal2) // 2

        min_idx = np.argmin(signal2[2:half])
        max_idx = np.argmax(signal2[2:half])
        assert min_idx <= max_idx

        time = time2[min_idx:max_idx + 1]
        signal1 = signal1[min_idx:max_idx + 1]
        signal2 = signal2[min_idx:max_idx + 1]

        return time, signal1, signal2

    def save_waveform(self, time_axis, data_points, filename):
        if time_axis is None or data_points is None:
            print("No waveform data to save.")
            return

        waveform_data = {"time": time_axis.tolist(), "voltage": data_points.tolist()}
        try:
            with open(filename, "w") as json_file:
                json.dump(waveform_data, json_file, indent=4)
            print(f"Waveform saved to {filename}")
        except Exception as e:
            print("Error saving waveform:", e)

    def plot_waveform(self, time_axis, data_points):
        if time_axis is None or data_points is None:
            print("No waveform data to plot.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data_points, label=f"{self.channel} Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Rigol DS1074Z Waveform")
        plt.legend()
        plt.grid()
        plt.show()

    def close(self):
        if self.oscilloscope:
            self.oscilloscope.close()
            print("Connection closed.")

    def plot_fft(self, time, signal, title="FFT of Cavity Transmission"):
        # Remove DC offset (baseline removal)
        signal = signal - np.mean(signal)

        # Time interval between samples
        dt = time[1] - time[0]

        # Fourier transform
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), d=dt)

        # View positive frequencies only
        half = len(signal) // 2
        fft_freq = fft_freq[:half]
        fft_magnitude = np.abs(fft_result[:half]) ** 2  # power spectrum

        plt.figure(figsize=(10, 4))
        plt.plot(fft_freq, fft_magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(title)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    osc = RigolOscilloscope()

    print("Connecting..")
    osc.connect()
    print("Connected Done")

    time_axis, data_points = osc.get_trim_waveform()
    osc.plot_waveform(time_axis, data_points)
    osc.plot_fft(time_axis, data_points)

    osc.close()
    print("Closed")
