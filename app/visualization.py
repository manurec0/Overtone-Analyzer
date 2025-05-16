import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer, Qt

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

class Visualization(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = "Waveform"
        self.audio_manager = None
        self.player = None
        self.analysis_engine = None
        self.is_recording = False

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('black')
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('bottom', 'Time', units='s')
        self.plot_item.setLabel('left', 'Amplitude')
        self.waveform_curve = self.plot_item.plot(pen=pg.mkPen('red', width=1))

        self.playhead = Playhead(self.plot_item)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)

        self.plot_item.vb.sigXRangeChanged.connect(self.limit_x_to_positive)
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.setMouseEnabled(x=True, y=False)

    def start_loop(self, audio_manager, player, analysis_engine):
        self.audio_manager = audio_manager
        self.player = player
        self.analysis_engine = analysis_engine
        self.playhead.player = self.player
        self.timer.start(16)

    def plot_waveform(self, data, duration):
        if data.ndim > 1:
            data = data.mean(axis=1)
        times = np.linspace(0, duration, len(data))
        self.waveform_curve.setData(times, data, pen='w')
        self.plot_widget.setXRange(0, duration, padding=0)

    def plot_live_waveform(self, data):
        duration = len(data) / self.audio_manager.get_samplerate()
        times = np.linspace(0, duration, len(data))
        self.waveform_curve.setData(times, data, pen=pg.mkPen('darkred', width=1))
        # Scroll logic
        window_width = 5
        if duration > window_width:
            self.plot_widget.setXRange(duration - window_width, duration, padding=0)
        else:
            self.plot_widget.setXRange(0, window_width, padding=0)

    def limit_x_to_positive(self, view_box):
        x_min, x_max = view_box.viewRange()[0]
        if x_min < 0:
            view_box.setXRange(0, x_max - x_min, padding=0)

    def update_loop(self):
        if self.is_recording:
            data = self.audio_manager.get_recording_data()
            if len(data) > 0:
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
        self.waveform_curve.setPen(pg.mkPen('white', width=1))

    def set_mode(self, mode):
        self.mode = mode
