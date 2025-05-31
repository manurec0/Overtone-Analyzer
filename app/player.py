import vlc
import numpy as np
import threading
import pyaudio
import time


class Player:
    def __init__(self):
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.loaded = False
        self.reached_end = False
        self.file_path = None
        self.needs_reload = False

        self.buffer_data = None
        self.buffer_loaded = False
        self.buffer_is_playing = False
        self.buffer_is_paused = False
        self.buffer_pos = 0
        self.buffer_thread = None
        self.buffer_stop_flag = threading.Event()
        self.buffer_pause_event = threading.Event()
        self.buffer_pause_event.set()
        self.buffer_samplerate = 44100
        self.stop_requested = False  # 🔁 new flag

        self.p = pyaudio.PyAudio()

        events = self.media_player.event_manager()
        events.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_end_reached)

    def on_end_reached(self, event):
        print("⚡ Media reached end. Scheduling reload.")
        self.needs_reload = True

    def reload_file(self):
        if self.file_path:
            print(f"✅ Reloading file safely: {self.file_path}")
            self.load_file(self.file_path)

    def load_file(self, file_path):
        self.stop()  # Will force full VLC reset now

        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        media = self.instance.media_new(file_path)
        self.media_player.set_media(media)

        events = self.media_player.event_manager()
        events.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_end_reached)

        self.file_path = file_path
        self.loaded = True
        self.reached_end = False
        self.needs_reload = False
        self.unload_buffer()

    def load_buffer(self, buffer_data, samplerate):
        self.stop()
        self.buffer_data = np.clip(buffer_data, -1.0, 1.0)
        self.buffer_samplerate = samplerate
        self.buffer_loaded = True
        self.buffer_is_playing = False
        self.buffer_is_paused = False
        self.buffer_pos = 0
        self.loaded = False
        self.file_path = None
        print(f"✅ Recorded buffer loaded | Duration: {len(self.buffer_data) / self.buffer_samplerate:.2f}s")

    def unload_buffer(self):
        self.stop()
        self.buffer_data = None
        self.buffer_loaded = False
        self.buffer_is_playing = False
        self.buffer_is_paused = False
        self.buffer_pos = 0

    def _buffer_playback_loop(self):
        if self.stop_requested or self.buffer_data is None:
            print("⚠️ Playback aborted before starting")
            return

        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.buffer_samplerate,
            output=True,
            frames_per_buffer=1024
        )

        self.buffer_is_playing = True
        self.stop_requested = False

        while True:
            self.buffer_pause_event.wait()

            # Safeguard in case buffer_data gets cleared mid-playback
            if self.buffer_data is None:
                print("⚠️ Buffer data was cleared mid-playback. Exiting cleanly.")
                break

            if self.buffer_stop_flag.is_set() or self.stop_requested:
                break

            chunk_size = 1024
            if self.buffer_pos >= len(self.buffer_data):
                break

            end_pos = min(self.buffer_pos + chunk_size, len(self.buffer_data))
            chunk = self.buffer_data[self.buffer_pos:end_pos]
            stream.write(chunk.astype(np.float32).tobytes())
            self.buffer_pos = end_pos

        stream.stop_stream()
        stream.close()
        self.buffer_is_playing = False
        self.buffer_pos = 0
        print("✅ Finished playing recorded buffer. Will allow replay.")

    def play(self):
        if self.buffer_loaded:
            if self.buffer_is_playing and self.buffer_is_paused:
                print("▶️ Resuming buffer playback")
                self.buffer_is_paused = False
                self.buffer_pause_event.set()
            elif not self.buffer_is_playing or not (self.buffer_thread and self.buffer_thread.is_alive()):
                if self.buffer_thread and self.buffer_thread.is_alive():
                    print("⚠️ Previous buffer thread still running. Forcing stop...")
                    self.stop()

                if self.buffer_pos >= len(self.buffer_data):  # ✅ MUST be before starting
                    print("↩️ Rewinding buffer before playback")
                    self.buffer_pos = 0

                print("▶️ Starting buffer playback")
                self.stop_requested = False
                self.buffer_stop_flag.clear()
                self.buffer_pause_event.set()
                self.buffer_thread = threading.Thread(target=self._buffer_playback_loop)
                self.buffer_thread.start()

    def pause(self):
        if self.buffer_loaded:
            if self.buffer_is_playing:
                print("⏸ Pausing buffer playback")
                self.buffer_is_paused = True
                self.buffer_pause_event.clear()
        elif self.loaded:
            self.media_player.pause()

    def stop(self):
        if self.buffer_loaded:
            self.buffer_stop_flag.set()
            self.stop_requested = True

            if self.buffer_thread and self.buffer_thread.is_alive():
                print("🔁 Joining buffer thread...")
                self.buffer_thread.join(timeout=2.0)
                if self.buffer_thread.is_alive():
                    print("⚠️ Buffer thread still alive after join")
                else:
                    print("🔁 Buffer thread successfully joined")

            self.buffer_thread = None  # ✅ Fully clear
            self.buffer_stop_flag.clear()  # ✅ Reset for next run
            self.buffer_pause_event.set()  # ✅ Unpause to allow exit
            self.buffer_pos = 0
            self.buffer_is_playing = False
            self.buffer_is_paused = False

    def get_time(self):
        if self.buffer_loaded:
            return self.buffer_pos / self.buffer_samplerate
        elif self.loaded:
            return self.media_player.get_time() / 1000.0
        return 0.0

    def get_duration(self):
        if self.buffer_loaded:
            return len(self.buffer_data) / self.buffer_samplerate
        elif self.loaded:
            return self.media_player.get_length() / 1000.0
        return 0.0

    def set_time(self, time_s):
        if self.buffer_loaded:
            self.buffer_pos = int(time_s * self.buffer_samplerate)
            self.buffer_pos = max(0, min(self.buffer_pos, len(self.buffer_data)))
        elif self.loaded:
            self.media_player.set_time(int(time_s * 1000.0))
            self.reached_end = False

    def is_playing(self):
        if self.buffer_loaded:
            return self.buffer_is_playing and not self.buffer_is_paused
        elif self.loaded:
            return self.media_player.is_playing() == 1
        return False

    def has_data(self):
        return self.buffer_loaded or self.loaded

    @property
    def time(self):
        return self.get_time()

