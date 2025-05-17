import numpy as np
import pysptk
from utils.helpers import freq_to_note_name, freq_to_note_index

class AnalysisEngine:
    def __init__(self, frame_length=2048, hopsize=256, rate=44100):
        self.frame_length = frame_length
        self.hopsize = hopsize
        self.rate = rate

    def detect_pitch(self, signal):
        if len(signal) < self.frame_length:
            return None
        frame = signal[-self.frame_length:]
        f0 = pysptk.swipe(frame.astype(np.float32), fs=self.rate, hopsize=self.hopsize, otype="f0")
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
                note_index = freq_to_note_index(f0)
                if note_index is not None:
                    times.append(i / self.rate)
                    note_indices.append(note_index)

        print(f"✅ Full pitch detection finished. {len(times)} valid notes.")
        return np.array(times), np.array(note_indices)
