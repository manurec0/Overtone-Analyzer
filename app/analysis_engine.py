import pyaudio
import soundfile as sf
import numpy as np

class AnalysisEngine:
    def __init__(self, rate=44100, chunk=4096):
        self.rate = rate
        self.chunk = chunk
        self.mic_stream = None
        self.wav_data = None
        self.wav_position = 0
        self.source = "wav"
        self.playing = False

    def load_wav(self, file_path):
        data, samplerate = sf.read(file_path, dtype='float32', always_2d=True)
        self.wav_data = data[:, 0]
        self.wav_position = 0
        self.rate = samplerate
        return len(self.wav_data) / self.rate

    def start_mic(self):
        if self.mic_stream is None:
            self.mic_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1,
                                                     rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.source = "mic"

    def stop_mic(self):
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        self.source = "wav"

    def get_next_chunk(self):
        if self.source == "mic" and self.mic_stream:
            data = self.mic_stream.read(self.chunk, exception_on_overflow=False)
            return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif self.source == "wav" and self.wav_data is not None:
            idx = int(self.wav_position * self.rate)
            chunk = self.wav_data[idx:idx + self.chunk]
            self.wav_position += self.chunk / self.rate
            return np.pad(chunk, (0, self.chunk - len(chunk))) if len(chunk) < self.chunk else chunk
        else:
            return np.zeros(self.chunk, dtype=np.float32)
