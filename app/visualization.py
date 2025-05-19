import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer, Qt

from utils.helpers import NOTE_NAMES_FULL, freq_to_note_index

class Playhead:
    def __init__(self, plot_item, player=None):
        self.player = player
        self.line = pg.InfiniteLine(angle=90, pen=pg.mkPen('white', width=2), movable=True)
        self.line.setCursor(Qt.SizeHorCursor)
        plot_item.addItem(self.line)
        self.visible = True
        self._default_pen = pg.mkPen('white', width=2)
        self._hover_pen = pg.mkPen('red', width=4)

        self.line.sigPositionChangeFinished.connect(self.on_drag_finished)
        self.line.scene().sigMouseMoved.connect(self.on_hover)

    def update_position(self, current_time):
        if self.visible:
            self.line.setPos(current_time)

    def on_drag_finished(self):
        if self.player and self.player.loaded:
            new_time = self.line.value()
            duration = self.player.media_player.get_length() / 1000.0
            new_time = max(0, min(new_time, duration))
            self.player.set_time(new_time * 1000.0)

    def on_hover(self, pos):
        vb = self.line.getViewBox()
        if vb is not None:
            mouse_x = vb.mapSceneToView(pos).x()
            if abs(mouse_x - self.line.value()) < 0.1:
                self.line.setPen(self._hover_pen)
            else:
                self.line.setPen(self._default_pen)

# [...] All your existing imports and class declarations stay

class Visualization(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = "Waveform"
        self.audio_manager = None
        self.player = None
        self.analysis_engine = None
        self.is_recording = False

        self.waveform_cache = None
        self.pitch_detection_cache = None

        # 🔽 NEW live plotting buffers
        self.live_note_times = []
        self.live_note_indices = []
        self.max_live_duration = 5
        self.live_waveform_buffer = np.zeros(0, dtype=np.float32)
        self.live_time = 0.0
        self.frame_duration = 4096 / 44100  # Will update dynamically from audio_manager if needed

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('black')
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('bottom', 'Time', units='s')
        self.playhead = Playhead(self.plot_item)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)

        self.plot_item.vb.sigXRangeChanged.connect(self.limit_x_to_positive)
        self.plot_widget.setMouseEnabled(x=True, y=False)

        # Plot items
        self.waveform_curve = None
        self.pitch_scatter = None

    def start_loop(self, audio_manager, player, analysis_engine):
        self.audio_manager = audio_manager
        self.player = player
        self.analysis_engine = analysis_engine
        self.playhead.player = self.player
        self.timer.start(16)
        self.frame_duration = audio_manager.chunk_size / audio_manager.get_samplerate()

    def update_live_visualization(self, result):
        if self.mode == "Waveform":
            self.plot_live_waveform(result["signal"])

        elif self.mode == "Fundamental Pitch Detection":
            pitch = result["pitch"]
            if pitch:
                note_index = freq_to_note_index(pitch)
                if note_index is not None:
                    self.live_note_times.append(self.live_time)
                    self.live_note_indices.append(note_index)
                    if len(self.live_note_times) > 100:
                        self.live_note_times.pop(0)
                        self.live_note_indices.pop(0)
                    self.plot_live_piano_roll(self.live_note_times, self.live_note_indices)
            self.live_time += self.frame_duration

    def plot_live_piano_roll(self, times, indices):
        if self.pitch_scatter:
            self.pitch_scatter.setData(times, indices)
            self.plot_widget.setXRange(max(0, self.live_time - 5), self.live_time)

    def plot_waveform(self, data, duration):
        if self.waveform_cache is None:
            print("🔄 Processing and caching waveform...")
            if data.ndim > 1:
                data = data.mean(axis=1)
            times = np.linspace(0, duration, len(data))
            self.waveform_cache = (times, data)
        else:
            print("✅ Using cached waveform.")

        times, data = self.waveform_cache
        self.waveform_curve.setData(times, data, pen='w')
        self.plot_widget.setXRange(0, duration, padding=0)

    def plot_live_waveform(self, data):
        samplerate = self.audio_manager.get_samplerate()
        chunk_size = len(data)

        x = np.arange(0, chunk_size)
        self.waveform_curve.setData(x, data, pen=pg.mkPen('darkred', width=1))

        self.plot_widget.setXRange(0, chunk_size, padding=0)
        self.plot_widget.setYRange(-10000, 10000)  # or adjust based on signal scale

    def plot_full_pitch_detection(self, data):
        if self.pitch_detection_cache is None:
            print("🔄 Processing and caching pitch detection...")
            times, note_indices = self.analysis_engine.process_pitch_detection_full(data)
            self.pitch_detection_cache = (times, note_indices)
        else:
            print("✅ Using cached pitch detection.")

        times, note_indices = self.pitch_detection_cache
        if len(times) > 0:
            self.pitch_scatter.setData(times, note_indices)
            self.plot_widget.setXRange(0, times[-1], padding=0)
        else:
            print("⚠️ No pitches detected.")

    def limit_x_to_positive(self, view_box):
        x_min, x_max = view_box.viewRange()[0]
        if x_min < 0:
            view_box.setXRange(0, x_max - x_min, padding=0)

    def update_loop(self):
        if self.is_recording:
            data = self.audio_manager.get_recording_data()
            if len(data) > 0:
                if self.mode == "Waveform":
                    self.plot_live_waveform(data)
                duration = len(data) / self.audio_manager.get_samplerate()
                self.playhead.update_position(duration)
        elif self.player:
            if self.player.needs_reload:
                print("🔄 Reload triggered by end reached flag in main loop")
                self.player.reload_file()
            current_time = self.player.get_time()
            if self.player.is_playing():
                self.playhead.update_position(current_time)

    def start_recording_mode(self):
        self.is_recording = True
        self.set_mode("Waveform")

    def stop_recording_mode(self):
        self.is_recording = False
        if self.waveform_curve:
            self.waveform_curve.setPen(pg.mkPen('white', width=1))

    def set_mode(self, mode):
        self.mode = mode
        self.plot_item.clear()
        self.playhead = Playhead(self.plot_item, self.player)
        self.waveform_curve = None
        self.pitch_scatter = None

        # Reset live buffers
        self.live_note_times = []
        self.live_note_indices = []
        self.live_time = 0.0

        if mode == "Waveform":
            self.plot_item.setLabel('left', 'Amplitude')
            self.plot_widget.setYRange(-1, 1)
            self.plot_item.getAxis('left').setTicks([])
            self.plot_widget.setMouseEnabled(x=True, y=False)
            self.plot_item.showGrid(x=True, y=True)
            self.waveform_curve = self.plot_item.plot(pen='w')

            if self.audio_manager and self.audio_manager.wav_data is not None:
                self.plot_waveform(self.audio_manager.wav_data,
                                   len(self.audio_manager.wav_data) / self.audio_manager.get_samplerate())

        elif mode == "Fundamental Pitch Detection":
            note_positions = [(i, NOTE_NAMES_FULL[i]) for i in range(len(NOTE_NAMES_FULL))]
            self.plot_item.setLabel('left', 'Note')
            self.plot_widget.setYRange(0, len(NOTE_NAMES_FULL))
            self.plot_item.getAxis('left').setTicks([note_positions])

            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_item.showGrid(x=True, y=True)

            self.pitch_scatter = pg.ScatterPlotItem(size=12, brush='cyan')
            self.plot_item.addItem(self.pitch_scatter)

            if self.audio_manager and self.audio_manager.wav_data is not None:
                self.plot_full_pitch_detection(self.audio_manager.wav_data)

    def clear_caches(self):
        self.waveform_cache = None
        self.pitch_detection_cache = None

    def clear_visualization(self):
        if self.waveform_curve:
            self.waveform_curve.clear()
        if self.pitch_scatter:
            self.pitch_scatter.setData([], [])
        self.live_note_times = []
        self.live_note_indices = []
        self.live_time = 0.0

