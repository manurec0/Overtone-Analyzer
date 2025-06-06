import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6 import QtCore
import pandas as pd
import os

from utils.helpers import NOTE_NAMES_FULL, freq_to_note_index, freq_to_note_name, note_index_to_freq


class Playhead:
    def __init__(self, plot_item, player=None):
        self.player = player
        self.line = pg.InfiniteLine(angle=90, pen=pg.mkPen('white', width=2), movable=True)
        self.line.setCursor(Qt.SizeHorCursor)
        self.line.setZValue(1000)
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
        if not self.player:
            return

        new_time = self.line.value()

        if self.player.buffer_loaded:
            duration = self.player.get_duration()
            new_time = max(0, min(new_time, duration))
            self.player.set_time(new_time)
            print(f"⏪ Set buffer_pos to {self.player.buffer_pos} ({new_time:.2f}s)")

        elif self.player.loaded:
            duration = self.player.media_player.get_length() / 1000.0
            new_time = max(0, min(new_time, duration))
            self.player.set_time(new_time * 1000.0)

    def on_hover(self, pos):
        if self.line.scene() is None:
            return  # Avoid crash on init

        vb = self.line.getViewBox()
        if vb is not None:
            try:
                view_pos = vb.mapSceneToView(pos)
                mouse_x = view_pos.x()
                if abs(mouse_x - self.line.value()) < 0.1:
                    self.line.setPen(self._hover_pen)
                else:
                    self.line.setPen(self._default_pen)
            except Exception as e:
                # Catch and fallback (optional: log or print once)
                self.line.setPen(self._default_pen)


class Visualization(QWidget):

    # Signals
    plot_overtone_signal = Signal(object, object)  # (pitch, harmonics_info)
    live_status_update = Signal(str, bool)  # text, visible

    def __init__(self, app_state):
        super().__init__()
        self.detection_results = None
        self.mode = "Waveform"
        self.audio_manager = None
        self.player = None
        self.analysis_engine = None
        self.app_state = app_state
        self.is_recording = False

        self.waveform_cache = None
        self.pitch_detection_cache = None

        self.harmonic_scatter = None
        self.full_spectrogram_cache = None

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
        self.playhead.line.setZValue(100)  # Higher Z = always on top

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)

        self.plot_item.vb.sigXRangeChanged.connect(self.limit_x_to_positive)
        self.plot_widget.setMouseEnabled(x=True, y=False)

        # Plot items
        self.waveform_curve = None
        self.pitch_scatter = None

        # Spectrogram
        self.spectrogram_buffer = None
        self.spectrogram_img = None
        self.spectrogram_width = 200  # number of time frames
        self.spectrogram_height = 2048  # N_FFT // 2 + 1
        self.spectrogram_vmin = -90
        self.spectrogram_vmax = 50

        # Overtone Profile
        self.overtone_bars = []
        self.overtone_thresholds = []
        self.plot_overtone_signal.connect(self.plot_overtone_profile)

    def start_loop(self, audio_manager, player, analysis_engine):
        self.audio_manager = audio_manager
        self.player = player
        self.analysis_engine = analysis_engine
        self.playhead.player = self.player
        self.timer.start(16)
        self.frame_duration = audio_manager.chunk_size / audio_manager.get_samplerate()

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
                if self.audio_manager and self.audio_manager.wav_data is not None:
                    sr = self.audio_manager.get_samplerate()
                    current_sample = int(current_time * sr)

                    frame_size = self.analysis_engine.frame_length
                    start = max(0, current_sample - frame_size)
                    end = start + frame_size

                    signal = self.audio_manager.wav_data[start:end]
                    if len(signal) == frame_size:
                        result = self.analysis_engine.process_live_frame(signal)

                        if self.mode == "Waveform" and self.app_state.isOscilloscope:
                            self.plot_oscilloscope_waveform(result["signal"])
                        elif self.mode == "Overtone Profile":
                            pitch = result.get("pitch")
                            harmonics = result.get("harmonics")
                            self.plot_overtone_signal.emit(pitch, harmonics)
                        self.update_text_labels(result)

    def update_live_visualization(self, result):
        pitch = result["pitch"]

        if self.mode == "Waveform":
            self.plot_live_waveform(result["signal"])

        elif self.mode == "Fundamental Pitch Detection":
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

        elif self.mode == "Spectrogram":
            self.plot_spectrogram_frame(result["magnitude"])

        elif self.mode == "Overtone Profile":
            harmonics = result["harmonics"]
            self.plot_overtone_signal.emit(pitch, harmonics)

        elif self.mode == "Overtone Analyzer":
            freqs = result.get("freqs")
            magnitude = result.get("magnitude")
            # if freqs is not None and magnitude is not None:
                # self.plot_spectrogram_frame(magnitude, freqs)

            # Fundamental pitch
            if pitch:
                note_index = freq_to_note_index(pitch)
                if note_index is not None:
                    self.live_note_times.append(self.live_time)
                    self.live_note_indices.append(note_index)

            # Harmonic pitch
            h = result.get("active_harmonic")
            if h and pitch:
                harmonic_freq = h * pitch
                harmonic_index = freq_to_note_index(harmonic_freq)
                if harmonic_index is not None:
                    self.live_harmonic_times.append(self.live_time)
                    self.live_harmonic_indices.append(harmonic_index)

            # Set scatter data
            if self.pitch_scatter:
                self.pitch_scatter.setData(self.live_note_times, self.live_note_indices)
            if self.harmonic_scatter:
                self.harmonic_scatter.setData(self.live_harmonic_times, self.live_harmonic_indices)

            # Scroll X range
            self.plot_widget.setXRange(max(0, self.live_time - 5), self.live_time)
            self.live_time += self.frame_duration

        self.update_text_labels(result)

    def update_text_labels(self, result):
        pitch = result.get("pitch")

        if pitch:
            note = freq_to_note_name(pitch)
            pitch_label = f"Note: {note} | {pitch:.1f} Hz"
            self.live_status_update.emit(pitch_label, True)

            if result.get("active_harmonic") and result.get("harmonic_info"):
                harmonic_label = f"Active Harmonic: H{result['active_harmonic']} ({result['harmonic_info']})"
                self.live_status_update.emit(harmonic_label, False)
            else:
                self.live_status_update.emit("Active Harmonic: ...", False)
        else:
            self.live_status_update.emit("Note: ...", True)
            self.live_status_update.emit("Active Harmonic: ...", False)

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
        x, y = self.get_waveform_points(data)
        self.waveform_curve.setData(x, y, pen=pg.mkPen('darkred', width=1))
        self.plot_widget.setXRange(0, len(data), padding=0)
        self.plot_widget.setYRange(-1.5, 1.5) # adjust scale

    def plot_oscilloscope_waveform(self, signal):
        x, y = self.get_waveform_points(signal)
        self.waveform_curve.setData(x, y, pen=pg.mkPen('white', width=1))
        self.plot_widget.setXRange(0, len(signal), padding=0)
        self.plot_widget.setYRange(-1.5, 1.5)

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

            wav_path = self.audio_manager.get_filepath()
            wav_path = self.audio_manager.get_filepath()
            if wav_path:
                gt_dir = os.path.dirname(wav_path)
                f0_path = os.path.join(gt_dir, "fundamental_ground_truth.csv")

                if os.path.exists(f0_path):
                    try:
                        f0_df = self.analysis_engine.load_f0_ground_truth(f0_path)
                        print("📊 Evaluating fundamental pitch detection...")

                        detection_results = self.analysis_engine.format_single_detection_results(
                            times, note_indices, label="fundamental"
                        )

                        eval_metrics = self.analysis_engine.evaluate_fundamentals_only(f0_df, detection_results)
                        print("🎯 Evaluation Metrics (Fundamental):")
                        for key, value in eval_metrics.items():
                            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                    except Exception as e:
                        print(f"⚠️ Could not load or evaluate ground truth: {e}")
                else:
                    print("ℹ️ No ground truth CSV found for fundamental pitch.")

        else:
            print("⚠️ No pitches detected.")

    def plot_spectrogram_frame(self, magnitude, freqs=None):
        if self.spectrogram_buffer is None:
            return

        if self.mode == "Overtone Analyzer" and freqs is not None:
            column = np.full((len(NOTE_NAMES_FULL),), self.spectrogram_vmin, dtype=np.float32)
            magnitude = np.clip(magnitude, self.spectrogram_vmin, self.spectrogram_vmax)

            for i, mag in enumerate(magnitude):
                note_idx = freq_to_note_index(freqs[i])
                if note_idx is not None and 0 <= note_idx < len(NOTE_NAMES_FULL):
                    column[note_idx] = max(column[note_idx], mag)

            self.spectrogram_buffer[:, :-1] = self.spectrogram_buffer[:, 1:]
            self.spectrogram_buffer[:, -1] = column

            self.spectrogram_img.setImage(
                self.spectrogram_buffer,
                levels=(self.spectrogram_vmin, self.spectrogram_vmax)
            )

        else:
            # Default spectrogram logic for other modes
            magnitude = np.clip(magnitude, self.spectrogram_vmin, self.spectrogram_vmax)

            if len(magnitude) > self.spectrogram_height:
                magnitude = magnitude[:self.spectrogram_height]

            self.spectrogram_buffer[:, :-1] = self.spectrogram_buffer[:, 1:]
            self.spectrogram_buffer[:, -1] = magnitude

            self.spectrogram_img.setImage(
                self.spectrogram_buffer.copy(),
                levels=(self.spectrogram_vmin, self.spectrogram_vmax)
            )

    def plot_overtone_profile(self, pitch, harmonics_info):
        # Clear previous bars
        for item in self.overtone_bars + self.overtone_thresholds:
            self.plot_item.removeItem(item)

        self.overtone_bars.clear()
        self.overtone_thresholds.clear()

        if pitch is None or not harmonics_info:
            return

        harmonics = []
        mags = []
        threshold_mags = []

        base_threshold = 0  # dBFS
        decay = np.linspace(0, 15, 16)

        for h in harmonics_info:
            n = h["harmonic"]
            mag_db = h["magnitude"]  # ✅ already in dB
            harmonics.append(n)
            mags.append(mag_db)

            threshold = base_threshold - decay[n - 1]
            threshold_mags.append(threshold)

            delta = mag_db - threshold
            color = 'g' if delta > 10 else 'orange' if delta > 0 else 'r'
            bar = pg.BarGraphItem(x=[n], height=[max(0, delta)], width=0.8, brush=color, y0=threshold)
            self.overtone_bars.append(bar)
            self.plot_item.addItem(bar)

            threshold_bar = pg.BarGraphItem(x=[n], height=[threshold], width=0.8, brush=(150, 150, 150, 100))
            self.overtone_thresholds.append(threshold_bar)
            self.plot_item.addItem(threshold_bar)

    def limit_x_to_positive(self, view_box):
        x_min, x_max = view_box.viewRange()[0]
        if x_min < 0:
            view_box.setXRange(0, x_max - x_min, padding=0)

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
        self.plot_item.getAxis('left').setTicks([])  # Clear any prior labels
        self.plot_item.setLabel('left', '')
        self.plot_item.setLabel('bottom', '')

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
            self.plot_item.getAxis('left').setTicks([[(i, str(i)) for i in range(-2, 2)]])
            self.plot_widget.setMouseEnabled(x=True, y=False)
            self.plot_item.showGrid(x=True, y=True)

            self.plot_item.getAxis('bottom').setTicks(None)

            if self.app_state.isLive or self.app_state.isOscilloscope:
                self.plot_item.setLabel('bottom', 'Samples')
                self.show_playhead(False if self.app_state.isLive else True)
            else:
                self.plot_item.setLabel('bottom', 'Time (s)')
                self.show_playhead(True)

            if self.waveform_curve is None:
                self.waveform_curve = self.plot_item.plot(pen='w')  # Ensure it's created

            if self.audio_manager and self.audio_manager.wav_data is not None:
                self.plot_waveform(self.audio_manager.wav_data,
                                   len(self.audio_manager.wav_data) / self.audio_manager.get_samplerate())

        elif mode == "Fundamental Pitch Detection":
            self.show_playhead(True)
            note_positions = [(i, NOTE_NAMES_FULL[i]) for i in range(len(NOTE_NAMES_FULL))]
            self.plot_item.setLabel('left', 'Note')
            self.plot_item.setLabel('bottom', 'Time (s)')
            self.plot_item.setYRange(0, len(NOTE_NAMES_FULL))
            self.plot_item.setXRange(0, 5)
            self.plot_item.getAxis('left').setTicks([note_positions])

            self.plot_item.getAxis('bottom').setTicks(None)

            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_item.showGrid(x=True, y=True)

            self.pitch_scatter = pg.ScatterPlotItem(size=12, brush='cyan')
            self.plot_item.addItem(self.pitch_scatter)

            if self.audio_manager and self.audio_manager.wav_data is not None:
                self.plot_full_pitch_detection(self.audio_manager.wav_data)

        elif mode == "Spectrogram":
            self.plot_item.setLabel('left', 'Frequency (Hz)')
            self.plot_item.setLabel('bottom', 'Time (s)')
            self.plot_item.getAxis('left').setStyle(showValues=True)
            self.plot_item.showGrid(x=False, y=True)
            self.plot_widget.setMouseEnabled(x=False, y=True)
            self.plot_item.getAxis('bottom').setTicks(None)
            self.plot_item.getAxis('left').setTicks(None)

            samplerate = self.audio_manager.get_samplerate() if self.audio_manager and self.audio_manager.get_samplerate() else 44100
            nyquist = samplerate / 2

            self.spectrogram_img = pg.ImageItem()
            cmap = pg.colormap.get('inferno')  # or use 'plasma', 'viridis', etc.
            lut = cmap.getLookupTable(0.0, 1.0, 256)

            self.spectrogram_img.setLookupTable(lut)
            self.spectrogram_img.setOpts(axisOrder='row-major')
            self.spectrogram_img.setZValue(0)

            self.plot_item.addItem(self.spectrogram_img)
            if self.app_state.isLive:
                self.show_playhead(False)
                # Live mode: initialize scrolling buffer
                self.spectrogram_buffer = np.full(
                    (self.spectrogram_height, self.spectrogram_width),
                    self.spectrogram_vmin,
                    dtype=np.float32
                )

                duration = self.spectrogram_width * self.frame_duration

                self.plot_item.setXRange(0, duration)
                self.plot_item.setYRange(0, nyquist)
                self.spectrogram_img.setImage(
                    self.spectrogram_buffer,
                    levels=(self.spectrogram_vmin, self.spectrogram_vmax)
                )

                self.spectrogram_img.setRect(QtCore.QRectF(0, 0, duration, nyquist))

            elif self.audio_manager and self.audio_manager.wav_data is not None:
                self.show_playhead(True)
                if self.full_spectrogram_cache is None:
                    spec, freqs, times = self.analysis_engine.compute_full_spectrogram(
                        self.audio_manager.wav_data,
                        n_fft=4096,
                        hop_size=256
                    )

                    self.full_spectrogram_cache = (spec, freqs, times)

                else:
                    spec, freqs, times = self.full_spectrogram_cache

                self.spectrogram_buffer = spec
                self.spectrogram_img.setImage(
                    self.spectrogram_buffer,
                    levels=(self.spectrogram_vmin, self.spectrogram_vmax)
                )

                self.spectrogram_img.setRect(
                    QtCore.QRectF(0, 0, times[-1], freqs[-1])
                )

                self.plot_item.setXRange(0, times[-1])
                self.plot_item.setYRange(0, freqs[-1])

        elif mode == "Overtone Profile":
            self.show_playhead(False)
            self.plot_item.setLabel('left', 'Magnitude (dB)')
            self.plot_item.setLabel('bottom', 'Harmonic Number')
            self.plot_item.showGrid(x=True, y=True)
            self.plot_widget.setMouseEnabled(x=False, y=True)

            self.plot_item.setXRange(0.5, 16.5)
            self.plot_item.setYRange(0, 120)

            # Add Y-axis ticks every 10 dB
            yticks = [(i, f"{i}") for i in range(0, 130, 10)]
            self.plot_item.getAxis('left').setTicks([yticks])
            self.plot_item.getAxis('bottom').setTicks([[(i, f"H{i}") for i in range(1, 17)]])


        elif mode == "Overtone Analyzer":

            self.show_playhead(True)
            note_positions = [(i, NOTE_NAMES_FULL[i]) for i in range(len(NOTE_NAMES_FULL))]
            self.plot_item.setLabel('left', 'Note')
            self.plot_item.setLabel('bottom', 'Time (s)')
            self.plot_item.setYRange(0, len(NOTE_NAMES_FULL))
            self.plot_item.getAxis('left').setTicks([note_positions])
            self.plot_item.getAxis('bottom').setTicks(None)
            self.plot_item.showGrid(x=True, y=True)
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.spectrogram_img = pg.ImageItem()
            cmap = pg.colormap.get('inferno')
            self.spectrogram_img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
            self.spectrogram_img.setOpts(axisOrder='row-major')
            self.spectrogram_img.setZValue(0)
            self.plot_item.addItem(self.spectrogram_img)
            self.pitch_scatter = pg.ScatterPlotItem(size=10, brush='cyan')
            self.harmonic_scatter = pg.ScatterPlotItem(size=10, brush='green')
            self.plot_item.addItem(self.pitch_scatter)
            self.plot_item.addItem(self.harmonic_scatter)

            if self.app_state.isLive:
                self.pitch_scatter = pg.ScatterPlotItem(size=10, brush='cyan')
                self.harmonic_scatter = pg.ScatterPlotItem(size=10, brush='orange')
                self.plot_item.addItem(self.pitch_scatter)
                self.plot_item.addItem(self.harmonic_scatter)
                self.live_note_times = []
                self.live_note_indices = []
                self.live_harmonic_times = []
                self.live_harmonic_indices = []
                self.live_time = 0.0

            elif self.audio_manager and self.audio_manager.wav_data is not None:
                # Compute note-index spectrogram (as before)
                if self.full_spectrogram_cache is None:
                    spec, freqs, times = self.analysis_engine.compute_full_spectrogram(
                        self.audio_manager.wav_data,
                        n_fft=4096,
                        hop_size=256
                    )
                    self.full_spectrogram_cache = (spec, freqs, times)

                else:
                    spec, freqs, times = self.full_spectrogram_cache

                aligned_spec = np.full((len(NOTE_NAMES_FULL), spec.shape[1]),
                                       self.spectrogram_vmin, dtype=np.float32)
                for i, freq in enumerate(freqs):
                    note_idx = freq_to_note_index(freq)

                    if note_idx is not None and 0 <= note_idx < len(NOTE_NAMES_FULL):
                        aligned_spec[note_idx, :] = np.maximum(aligned_spec[note_idx, :], spec[i, :])

                self.spectrogram_buffer = aligned_spec
                self.spectrogram_img.setImage(aligned_spec, levels=(self.spectrogram_vmin, self.spectrogram_vmax))
                self.spectrogram_img.setRect(QtCore.QRectF(0, 0, times[-1], len(NOTE_NAMES_FULL)))
                self.plot_item.setXRange(0, times[-1])
                self.plot_item.setYRange(0, len(NOTE_NAMES_FULL))

                self.process_overtone_analyzer_full(self.audio_manager.wav_data)

    def clear_caches(self):
        self.waveform_cache = None
        self.pitch_detection_cache = None
        self.full_spectrogram_cache = None

    def clear_visualization(self):
        if self.waveform_curve:
            self.waveform_curve.clear()
        if self.pitch_scatter:
            self.pitch_scatter.setData([], [])
        self.live_note_times = []
        self.live_note_indices = []
        self.live_time = 0.0
        for item in self.overtone_bars + self.overtone_thresholds:
            self.plot_item.removeItem(item)
        self.overtone_bars.clear()
        self.overtone_thresholds.clear()

        self.set_mode(self.mode)

    import os

    def process_overtone_analyzer_full(self, data):
        print("🔍 Processing overtone analyzer view (file mode)...")

        sr = self.audio_manager.get_samplerate()
        frame_size = self.analysis_engine.frame_length
        hop_size = frame_size

        fund_times = []
        fund_indices = []
        harm_times = []
        harm_indices = []

        for i in range(0, len(data) - frame_size, hop_size):
            current_time = i / sr
            signal = data[i:i + frame_size]
            if len(signal) < frame_size:
                continue

            result = self.analysis_engine.process_live_frame(signal)
            pitch = result.get("pitch")

            if pitch:
                note_idx = freq_to_note_index(pitch)
                if note_idx is not None:
                    fund_times.append(current_time)
                    fund_indices.append(note_idx)

                h = result.get("active_harmonic")
                if h and h > 0:
                    harmonic_freq = pitch * h
                    harmonic_idx = freq_to_note_index(harmonic_freq)
                    if harmonic_idx is not None:
                        harm_times.append(current_time)
                        harm_indices.append(harmonic_idx)

        # Set plot data
        self.pitch_scatter.setData(fund_times, fund_indices)
        self.harmonic_scatter.setData(harm_times, harm_indices)

        if len(fund_times) > 0:
            self.plot_widget.setXRange(0, fund_times[-1], padding=0)

        print(f"✅ Plotted {len(fund_times)} fundamentals and {len(harm_times)} harmonics")

        # Store results
        self.detection_results = self.analysis_engine.format_detection_results(
            fund_times, fund_indices, harm_times, harm_indices,
            frame_duration=self.analysis_engine.frame_length / self.analysis_engine.rate
        )

        # Try to find ground truth files next to the loaded WAV
        wav_path = self.audio_manager.get_filepath()
        if not wav_path:
            print("⚠️ WAV file path not available, skipping evaluation.")
            return

        gt_dir = os.path.dirname(wav_path)
        f0_path = os.path.join(gt_dir, "fundamental_ground_truth.csv")
        harm_path = os.path.join(gt_dir, "harmonic_ground_truth.csv")

        if not (os.path.exists(f0_path) and os.path.exists(harm_path)):
            print("⚠️ Ground truth files not found in directory, skipping evaluation.")
            return

        # Load and evaluate
        f0_df, harmonics_df = self.analysis_engine.load_ground_truth(f0_path, harm_path)

        ground_truth_df = pd.concat([
            f0_df.assign(label="fundamental"),
            harmonics_df.assign(label="harmonic")
        ], ignore_index=True)

        # Run multi-type evaluation
        metrics_dict = self.analysis_engine.evaluate_mode(
            mode="overtone_analyzer",
            f0_df=ground_truth_df,
            detection_results=self.detection_results
        )

        # Print all evaluations
        for section, metrics in metrics_dict.items():
            print(f"\n🎯 Evaluation Metrics ({section.capitalize()}):")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    def show_playhead(self, show: bool):
        if show:
            self.playhead.line.show()
        else:
            self.playhead.line.hide()

    def get_waveform_points(self, signal):
        x = np.arange(len(signal))
        return x, signal
