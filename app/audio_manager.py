import numpy as np
import soundfile as sf
import pyaudio
import threading


class AudioManager:
    def __init__(self):
        self.source = None
        self.wav_data = None
        self.samplerate = 44100  # Fixed default samplerate
        self.chunk_size = 4096
        self.recording_buffer = np.zeros(0, dtype=np.float32)
        self.recording_lock = threading.Lock()

        self._pyaudio_instance = pyaudio.PyAudio()
        self._recording_stream = None
        self._stop_recording_flag = threading.Event()
        self._live_stream = None
        self.live_callback = None

    def load_wav(self, filepath):
        wav_data, samplerate = sf.read(filepath)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        self.wav_data = wav_data.astype(np.float32)
        self.samplerate = samplerate
        self.source = "file"
        print(f"Loaded WAV file: {filepath} | Duration: {len(self.wav_data) / self.samplerate:.2f}s | SampleRate: {samplerate}Hz")
        return len(self.wav_data) / self.samplerate

    def unload_wav(self):
        self.wav_data = None
        self.samplerate = None
        self.source = None
        print("📤 WAV file unloaded.")

    def start_mic(self):
        self.stop_mic()
        self.unload_wav()
        self.recording_buffer = np.zeros(0, dtype=np.float32)
        self.source = "mic"
        self._stop_recording_flag.clear()

        def callback(in_data, frame_count, time_info, status):
            chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            with self.recording_lock:
                self.recording_buffer = np.concatenate((self.recording_buffer, chunk))
            return (in_data, pyaudio.paContinue)

        self._recording_stream = self._pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        self._recording_stream.start_stream()
        print("🎙️ Mic recording started")

    def stop_mic(self):
        self._stop_recording_flag.set()
        if self._recording_stream:
            self._recording_stream.stop_stream()
            self._recording_stream.close()
            self._recording_stream = None
            print("🛑 Mic recording stopped")
        self.source = None

    def get_recording_data(self):
        with self.recording_lock:
            return np.copy(self.recording_buffer)

    def get_samplerate(self):
        return self.samplerate

    def start_live_mode(self, callback):
        self.stop_current_stream()
        self.live_callback = callback
        self.source = "mic"

        self._live_stream = self._pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._live_stream_callback
        )
        self._live_stream.start_stream()
        print("🎧 Live Mode stream started")

    def _live_stream_callback(self, in_data, frame_count, time_info, status):
        if self.live_callback:
            audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            self.live_callback(audio_array)
        return (None, pyaudio.paContinue)

    def stop_current_stream(self):
        if self._live_stream:
            self._live_stream.stop_stream()
            self._live_stream.close()
            self._live_stream = None
            print("🛑 Live stream stopped")
        self.live_callback = None
