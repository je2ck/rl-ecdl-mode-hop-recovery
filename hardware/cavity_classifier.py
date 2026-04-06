import numpy as np
import matplotlib.pyplot as plt
from .oscilloscope import RigolOscilloscope

FSR = 2.5


class WaveformClassifier:
    def __init__(self, prominence=0.015, peak_threshold=3, data_threshold=0.035):
        """
        Initialize classifier with peak prominence and threshold.
        :param prominence: Minimum prominence to detect significant peaks.
        :param peak_threshold: Number of peaks required to classify as unstable.
        """
        self.prominence = prominence
        self.peak_threshold = peak_threshold
        self.data_threshold = data_threshold
        self.time = None
        self.data = None
        self.peaks = None

    def classify(self, time_axis, data_points, voltage, frequency=None, plotter=None):
        """
        Classifies waveform stability based on peak detection.
        :param time_axis: List of time values.
        :param data_points: List of voltage values.
        :return: "Stable" or "Unstable"
        """
        time_axis = np.array(time_axis, dtype=float)
        data_points = np.array(data_points, dtype=float)
        positive_mask = data_points > 0.0

        self.time = time_axis[positive_mask]
        self.data = data_points[positive_mask]

        if self.data.size == 0:
            self.data_threshold = 0.0
        else:
            self.data_threshold = np.max(self.data) * 0.2

        candidate_mask = self.data > self.data_threshold
        self.peaks = np.where(candidate_mask)[0]

        self.peaks = self.cluster_peaks(self.peaks)
        voltage_clustered = voltage[self.peaks]
        spacing = np.mean(np.diff(voltage_clustered))
        stable = "Unstable" if not (self.is_equally_spaced(self.peaks) and abs(spacing - FSR) < 1.0) else "Stable"

        if plotter is not None:
            plotter.update(self.time, self.data, self.peaks, frequency, voltage, footer_text=stable)
        return stable

    def cluster_peaks(self, peaks, distance=5):
        """
        Group nearby peaks and return one representative per group.
        """
        if len(peaks) == 0:
            return []

        clustered = []
        group = [peaks[0]]

        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i - 1] <= distance:
                group.append(peaks[i])
            else:
                clustered.append(int(np.mean(group)))
                group = [peaks[i]]
        clustered.append(int(np.mean(group)))

        return np.array(clustered)

    def is_equally_spaced(self, peaks, tolerance=0.1, verbose=False):
        """
        Check if the peaks are equally spaced within a certain tolerance.
        """
        if len(peaks) == 0:
            return False
        if len(peaks) < 3:
            return True

        spacings = np.diff(peaks)
        mean_spacing = np.mean(spacings)
        max_deviation = np.max(np.abs(spacings - mean_spacing) / mean_spacing)

        if verbose:
            print("Spacings:", spacings)
            print("Mean spacing:", mean_spacing)
            print("Max relative deviation:", max_deviation)

        return max_deviation <= tolerance

    def plot_waveform_with_peaks(self, time, data, peaks, title="Waveform with Peaks"):
        plt.figure(figsize=(12, 5))
        plt.plot(time, data, label="Signal")
        plt.plot(time[peaks], data[peaks], 'ro', label="Detected Peaks")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    osc = RigolOscilloscope()

    osc.connect()
    time, data, volt = osc.get_trim_waveform()
    print(WaveformClassifier().classify(time, data, volt))
    osc.close()
