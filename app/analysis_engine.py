import numpy as np
import pysptk
import utils.helpers as helpers
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from app.app_manager import AppState


class AnalysisEngine:
    def __init__(self, app_state, frame_length=4096, hopsize=256, rate=44100):
        self.frame_length = frame_length
        self.hopsize = hopsize
        self.rate = rate
        self.app_state = app_state

    def detect_pitch(self, signal, min_pitch=20.0, max_pitch=700.0):
        if len(signal) < self.frame_length:
            return None
        frame = signal[-self.frame_length:]
        f0 = pysptk.swipe(frame.astype(np.float32),
                          fs=self.rate,
                          hopsize=self.hopsize,
                          min=min_pitch,
                          max=max_pitch,
                          otype="f0")
        return next((p for p in f0 if p > 0), None)

    def detect_pitch_yin(self, signal: np.ndarray, sr: int) -> float | None:
        signal = signal.astype(np.float64)
        signal -= np.mean(signal)
        w_size = len(signal) // 2
        taus = np.arange(1, w_size)
        diff = np.array([np.sum((signal[:-t] - signal[t:]) ** 2) for t in taus])
        cmndf = diff[1:] / (np.cumsum(diff)[1:] / np.arange(1, len(diff)))
        threshold = 0.1
        candidates = np.where(cmndf < threshold)[0]
        if candidates.size == 0:
            return None
        tau = candidates[0]
        if 1 < tau < len(cmndf) - 1:
            y1, y2, y3 = cmndf[tau - 1:tau + 2]
            denom = 2 * (y1 - 2 * y2 + y3)
            if denom != 0:
                tau += (y1 - y3) / denom
        pitch = sr / (tau + 1)
        return pitch if 20 <= pitch <= 700 else None

    def detect_pitch_hps(self, signal: np.ndarray, rate: int = 44100, n_fft: int = 4096,
                         max_downsample: int = 4) -> float | None:
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        windowed = signal * np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        spectrum[spectrum == 0] = 1e-12  # Avoid log(0)

        hps = spectrum.copy()
        for h in range(2, max_downsample + 1):
            decimated = spectrum[::h]
            hps[:len(decimated)] *= decimated

        freqs = np.fft.rfftfreq(n_fft, 1 / rate)
        peak_idx = np.argmax(hps[:int(1000 * n_fft / rate)])
        pitch = freqs[peak_idx]

        return pitch if 20 <= pitch <= 700 else None

    def process_pitch_detection_full(self, data):
        times = []
        note_indices = []

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.clip(data, -1.0, 1.0)

        algo = self.app_state.pitch_algorithm  # Access current algorithm

        for i in range(0, len(data) - self.frame_length, self.frame_length):
            chunk = data[i:i + self.frame_length]

            # Select pitch detection method
            if algo == "YIN":
                f0 = self.detect_pitch_yin(chunk, self.rate)
            elif algo == "HPS":
                f0 = self.detect_pitch_hps(chunk, self.rate)
            elif algo == "CREPE":
                print("⚠️ CREPE not implemented yet.")
                f0 = None
            else:
                f0 = self.detect_pitch(chunk)

            if f0 and f0 > 0:
                note_index = helpers.freq_to_note_index(f0)
                if note_index is not None:
                    times.append(i / self.rate)
                    note_indices.append(note_index)

        print(f"✅ Full pitch detection ({algo}) finished. {len(times)} valid notes.")
        return np.array(times), np.array(note_indices)

    @staticmethod
    def smooth_spectrum(mag, window_size=5):
        return np.convolve(mag, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def parabolic_peak(mag, idx):
        if 1 <= idx < len(mag) - 1:
            alpha = mag[idx - 1]
            beta = mag[idx]
            gamma = mag[idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-6)
            return beta - 0.25 * (alpha - gamma) * p
        return mag[idx]

    def detect_active_harmonic_dynamic(self, freqs, magnitude, fundamental, max_harmonic=16):
        if fundamental is None:
            return None, None

        best_harmonic = None
        best_magnitude = -np.inf

        for n in range(4, max_harmonic + 1):  # Skip H1
            harmonic_freq = n * fundamental
            if harmonic_freq >= self.rate / 2:
                continue

            idx = np.argmin(np.abs(freqs - harmonic_freq))
            peak_mag = self.parabolic_peak(magnitude, idx)

            # Dynamic thresholds (you can adjust if needed)
            if n <= 6:
                threshold = 30
            elif n <= 12:
                threshold = 20
            else:
                threshold = 10

            if peak_mag > threshold and peak_mag > best_magnitude:
                best_magnitude = peak_mag
                best_harmonic = n

        return best_harmonic, best_magnitude

    def get_harmonic_info(self, fundamental, harmonic_number):
        freq = fundamental * harmonic_number
        note = helpers.freq_to_note_name(freq)
        intervals = {
            1: "Unison", 2: "Octave", 3: "Octave + Fifth", 4: "2 Octaves",
            5: "2 Octaves + Major Third", 6: "2 Octaves + Fifth",
            7: "2 Octaves + Minor Seventh", 8: "3 Octaves",
            9: "3 Octaves + Major Second", 10: "3 Octaves + Major Third",
            11: "3 Octaves + Augmented Fourth", 12: "3 Octaves + Fifth",
            13: "3 Octaves + Minor Sixth", 14: "3 Octaves + Minor Seventh",
            15: "3 Octaves + Major Seventh", 16: "4 Octaves"
        }
        interval = intervals.get(harmonic_number, f"{harmonic_number}th Harmonic")
        return f"{note} ({interval})"

    def process_live_frame(self, audio_chunk):
        signal = audio_chunk.astype(np.float32)
        peak = np.max(np.abs(signal)) + 1e-6
        signal = signal / peak

        algo = self.app_state.pitch_algorithm  # Read from global state

        # Use selected algorithm
        if algo == "YIN":
            pitch = self.detect_pitch_yin(signal, self.rate)
        elif algo == "HPS":
            pitch = self.detect_pitch_hps(signal, self.rate)
        elif algo == "CREPE":
            print("⚠️ CREPE not implemented yet.")
            pitch = None
        else:  # Default SWIPE
            pitch = self.detect_pitch(signal)

        window = np.hanning(len(signal))
        fft_size = self.frame_length * 2
        fft_result = np.fft.rfft(signal * window, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1 / self.rate)
        magnitude = 20 * np.log10(np.abs(fft_result) + 1e-6)
        magnitude = self.smooth_spectrum(magnitude)

        harmonics_info = []
        active_harmonic = None
        harmonic_info_str = None

        if pitch:
            for n in range(1, 17):
                harmonic_freq = pitch * n
                if harmonic_freq > self.rate / 2:
                    break
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonics_info.append({
                    "harmonic": n,
                    "freq": freqs[idx],
                    "magnitude": magnitude[idx]
                })

            best_h, best_mag = self.detect_active_harmonic_dynamic(freqs, magnitude, pitch)
            if best_h:
                active_harmonic = best_h
                harmonic_info_str = self.get_harmonic_info(pitch, best_h)

        return {
            "signal": signal,
            "pitch": pitch,
            "freqs": freqs,
            "magnitude": magnitude,
            "harmonics": harmonics_info,
            "active_harmonic": active_harmonic,
            "harmonic_info": harmonic_info_str
        }

    def compute_full_spectrogram(self, data, n_fft=4096, hop_size=256):
        if data.ndim > 1:
            data = data.mean(axis=1)  # Convert to mono

        data = data.astype(np.float32)
        peak = np.max(np.abs(data)) + 1e-6
        data = data / peak  # Normalize

        n_frames = 1 + (len(data) - n_fft) // hop_size
        spectrogram = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)

        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop_size
            end = start + n_fft
            frame = data[start:end]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))

            fft_result = np.fft.rfft(frame * window)
            magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-6)
            spectrogram[:, i] = magnitude_db

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.rate)
        times = np.arange(n_frames) * hop_size / self.rate

        print(f"✅ Computed spectrogram: shape = {spectrogram.shape}")
        return spectrogram, freqs, times

    @staticmethod
    def load_ground_truth(f0_path, harmonics_path):
        """Load Sonic Visualiser CSVs and compute end times for plotting."""
        f0_df = pd.read_csv(f0_path, header=None)
        harmonics_df = pd.read_csv(harmonics_path, header=None)

        # Assign expected column names
        f0_df.columns = ["time", "frequency", "duration", "confidence", "label"]
        harmonics_df.columns = ["time", "frequency", "duration", "confidence", "label"]

        # Convert frequency to float (handle commas)
        f0_df["frequency"] = f0_df["frequency"].astype(str).str.replace(",", ".").astype(float)
        harmonics_df["frequency"] = harmonics_df["frequency"].astype(str).str.replace(",", ".").astype(float)

        # Calculate end times for horizontal segments
        f0_df["end_time"] = f0_df["time"] + f0_df["duration"]
        harmonics_df["end_time"] = harmonics_df["time"] + harmonics_df["duration"]

        return f0_df, harmonics_df

    def plot_ground_truth(f0_df, harmonics_df):
        """Plot ground truth fundamental and harmonic annotations with duration."""
        plt.figure(figsize=(14, 6))

        # Plot F0 segments
        for _, row in f0_df.iterrows():
            plt.hlines(row["frequency"], row["time"], row["end_time"],
                       colors='cyan', linewidth=3, label='Fundamental (GT)')

        # Plot harmonic segments
        for _, row in harmonics_df.iterrows():
            plt.hlines(row["frequency"], row["time"], row["end_time"],
                       colors='green', linewidth=2, label='Harmonic (GT)')

        # Deduplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Ground Truth: Fundamental and Harmonics with Duration")
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def format_overtone_results(fund_times, fund_indices, harm_times, harm_indices, frame_duration=4096 / 44100):
        results = []

        for t, idx in zip(fund_times, fund_indices):
            freq = helpers.note_index_to_freq(idx)
            results.append({
                "type": "fundamental",
                "start_time": t,
                "end_time": t + frame_duration,
                "frequency": freq
            })

        for t, idx in zip(harm_times, harm_indices):
            freq = helpers.note_index_to_freq(idx)
            results.append({
                "type": "harmonic",
                "start_time": t,
                "end_time": t + frame_duration,
                "frequency": freq
            })

        return results

    def evaluate_results(self, ground_truth_df, detection_results,
                         freq_tolerance=20.0, min_time_overlap=0.3):
        from collections import defaultdict

        gt_matched = set()
        result_matched = set()

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        freq_errors = []

        gt_segments = ground_truth_df.to_dict("records")

        for i, det in enumerate(detection_results):
            matched = False
            for j, gt in enumerate(gt_segments):
                if gt["label"].lower().startswith(det["type"]):
                    # Check frequency match
                    if abs(gt["frequency"] - det["frequency"]) <= freq_tolerance:
                        # Check time overlap
                        start = max(gt["time"], det["start_time"])
                        end = min(gt["end_time"], det["end_time"])
                        overlap = max(0.0, end - start)
                        gt_duration = gt["end_time"] - gt["time"]
                        det_duration = det["end_time"] - det["start_time"]
                        if (overlap / gt_duration >= min_time_overlap or
                                overlap / det_duration >= min_time_overlap):
                            true_positives += 1
                            freq_errors.append(abs(gt["frequency"] - det["frequency"]))
                            gt_matched.add(j)
                            result_matched.add(i)
                            matched = True
                            break
            if not matched:
                false_positives += 1

        false_negatives = len(gt_segments) - len(gt_matched)

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_freq_error": np.mean(freq_errors) if freq_errors else None
        }

