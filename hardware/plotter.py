import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


class PlotManager:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_time, self.ax_fft) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        self.line_signal, = self.ax_time.plot([], [], label="Signal")
        self.line_peaks, = self.ax_time.plot([], [], 'ro', label="Detected Peaks")
        self.line_fft, = self.ax_fft.plot([], [], label="FFT")

        # Bottom text area
        self.textbox = self.fig.text(0.5, 0.01, 'Text', ha='center', fontsize=10, wrap=True)

        # Time domain plot
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_title("Waveform with Peaks")
        self.ax_time.grid(True)
        self.ax_time.legend()

        # Frequency domain plot
        self.ax_fft.set_xlabel("Time / Frequency (Hz)")
        self.ax_fft.set_ylabel("Power")
        self.ax_fft.set_title("FFT")
        self.ax_fft.grid(True)

        self.fig.tight_layout(rect=[0, 0.05, 1, 1])  # Reserve space for bottom text

    def update(self, time, data, peaks, frequency, voltage, footer_text=""):
        # Time domain
        self.line_signal.set_data(time, data)
        self.line_peaks.set_data(time[peaks], data[peaks])
        self.ax_time.relim()
        self.ax_time.autoscale_view()

        # FFT domain
        signal = data - np.mean(data)
        dt = time[1] - time[0]
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), d=dt)
        half = len(signal) // 2
        fft_freq = fft_freq[:half]
        fft_magnitude = np.abs(fft_result[:half]) ** 2

        self.line_fft.set_data(fft_freq, fft_magnitude)
        self.ax_fft.relim()
        self.ax_fft.autoscale_view()

        spacing = np.mean(np.diff(voltage[peaks]))
        self.textbox.set_text(footer_text + f"\nFrequency: {frequency} THz" + f" \tSpacing: {spacing:.2f} V" + f" \tPeaks: {len(peaks)}")

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if "Unstable" in footer_text:
            os.makedirs("unstable_plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unstable_plots/unstable_{timestamp}.png"
            self.fig.savefig(filename, dpi=150)
        elif "Stable" in footer_text:
            os.makedirs("stable_plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stable_plots/stable_{timestamp}.png"
            self.fig.savefig(filename, dpi=150)
