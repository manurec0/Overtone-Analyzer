import numpy as np
import pysptk
import utils.helpers as helpers

class AnalysisEngine:
    def __init__(self, frame_length=4096, hopsize=256, rate=44100):
        self.frame_length = frame_length
        self.hopsize = hopsize
        self.rate = rate

    def detect_pitch(self, signal, min_pitch=20.0, max_pitch=2000.0):
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

    def process_pitch_detection_full(self, data):
        times = []
        note_indices = []

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.clip(data, -1.0, 1.0)

        for i in range(0, len(data) - self.frame_length, self.frame_length):
            chunk = data[i:i + self.frame_length]
            f0 = self.detect_pitch(chunk)
            if f0 and f0 > 0:
                note_index = helpers.freq_to_note_index(f0)
                if note_index is not None:
                    times.append(i / self.rate)
                    note_indices.append(note_index)

        print(f"✅ Full pitch detection finished. {len(times)} valid notes.")
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
                threshold = -60
            elif n <= 12:
                threshold = -50
            else:
                threshold = -40

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
                #mag = self.parabolic_peak(magnitude, idx) # remove parabolic interpolation
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
