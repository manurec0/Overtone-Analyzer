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

    def process_live_frame(self, audio_chunk):
        # Ensure float32 for consistency
        signal = audio_chunk.astype(np.float32)

        # --- 1. Pitch Detection ---
        pitch = self.detect_pitch(signal)

        # --- 2. FFT Analysis ---
        window = np.hanning(len(signal))
        fft_result = np.fft.rfft(signal * window, n=self.frame_length * 2)
        freqs = np.fft.rfftfreq(len(fft_result), 1 / self.rate)
        magnitude = 20 * np.log10(np.abs(fft_result) + 1e-6)

        # --- 3. Harmonic Detection (Optional) ---
        harmonics_info = []
        if pitch:
            for n in range(1, 17):  # First 16 harmonics
                harmonic_freq = pitch * n
                if harmonic_freq > self.rate / 2:
                    break
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonics_info.append({
                    "harmonic": n,
                    "freq": freqs[idx],
                    "magnitude": magnitude[idx]
                })

        # --- Return structured result ---
        return {
            "signal": signal,
            "pitch": pitch,
            "freqs": freqs,
            "magnitude": magnitude,
            "harmonics": harmonics_info,
        }


