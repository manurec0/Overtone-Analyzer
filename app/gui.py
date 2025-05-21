from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                               QLineEdit, QLabel)
from PySide6.QtCore import Qt
from app.visualization import Visualization

class AppGUI(QMainWindow):
    def __init__(self, audio_manager, player, analysis_engine):
        super().__init__()

        self.isLive = False
        self.audio_manager = audio_manager
        self.player = player
        self.analysis_engine = analysis_engine

        self.setWindowTitle("Overtone Analyzer")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Bar Layout
        top_bar_layout = QHBoxLayout()
        controls_inner_layout = QHBoxLayout()
        controls_inner_layout.setAlignment(Qt.AlignCenter)

        self.rewind_start_btn = QPushButton("⏮", clicked=self.rewind_to_start)
        self.rewind_5s_btn = QPushButton("⏪ 5s", clicked=self.rewind_5s)
        self.play_pause_btn = QPushButton("▶️ Play", clicked=self.toggle_play)
        self.play_pause_btn.setEnabled(False)
        self.forward_5s_btn = QPushButton("⏩ 5s", clicked=self.forward_5s)
        self.forward_end_btn = QPushButton("⏭", clicked=self.forward_to_end)
        self.record_btn = QPushButton("🔴 Record", clicked=self.toggle_record)
        self.live_btn = QPushButton("Live Mode", clicked=self.toggle_live_mode)

        controls_inner_layout.addWidget(self.rewind_start_btn)
        controls_inner_layout.addWidget(self.rewind_5s_btn)
        controls_inner_layout.addWidget(self.play_pause_btn)
        controls_inner_layout.addWidget(self.forward_5s_btn)
        controls_inner_layout.addWidget(self.forward_end_btn)
        controls_inner_layout.addWidget(self.record_btn)

        top_bar_layout.addWidget(self.live_btn)

        top_bar_layout.addStretch()
        top_bar_layout.addLayout(controls_inner_layout)
        top_bar_layout.addStretch()
        main_layout.addLayout(top_bar_layout)

        # Visualization
        self.visualization = Visualization()
        main_layout.addWidget(self.visualization)
        self.visualization.live_status_update.connect(self.update_live_status_label)

        # Bottom Bar Layout
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        self.load_btn = QPushButton("Open File", clicked=self.load_wav)
        self.file_display = QLineEdit()
        self.file_display.setReadOnly(True)
        self.file_display.setPlaceholderText("No file selected")
        self.file_display.setMinimumWidth(500)

        bottom_layout.addWidget(self.load_btn)
        bottom_layout.addWidget(self.file_display)
        bottom_layout.addStretch()

        # Mode Drop Down Menu
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Waveform", "Fundamental Pitch Detection", "Spectrogram", "Overtone Profile"])
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        bottom_layout.addWidget(self.mode_selector)

        # Display Labels

        self.pitch_label = QLabel("Note: ...")
        self.harmonic_label = QLabel("Active Harmonic: ...")
        self.pitch_label.setStyleSheet("color: white; font-size: 14px;")
        self.harmonic_label.setStyleSheet("color: white; font-size: 14px;")
        main_layout.addWidget(self.pitch_label)
        main_layout.addWidget(self.harmonic_label)

    def load_wav(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV files (*.wav)")
        if filepath:
            duration = self.audio_manager.load_wav(filepath)
            self.player.load_buffer(self.audio_manager.wav_data, self.audio_manager.get_samplerate())
            self.visualization.set_mode(self.visualization.mode)  # Automatically plots depending on mode
            self.file_display.setText(filepath)
            self.play_pause_btn.setEnabled(True)
            self.play_pause_btn.setText("▶️ Play")

    def toggle_play(self):
        if self.player.has_data():
            if self.player.is_playing():
                self.player.pause()
                self.play_pause_btn.setText("▶️ Play")
            else:
                self.player.play()
                self.play_pause_btn.setText("⏸ Pause")

    def rewind_to_start(self):
        if self.player.has_data():
            self.player.set_time(0)
            self.visualization.playhead.update_position(0)

    def rewind_5s(self):
        if self.player.has_data():
            current = self.player.get_time()
            new_time = max(0, (current - 5))
            self.player.set_time(new_time)
            self.visualization.playhead.update_position(new_time)

    def forward_5s(self):
        if self.player.has_data():
            current = self.player.get_time()
            new_time = min(self.player.get_duration(), (current + 5))
            self.player.set_time(new_time)
            self.visualization.playhead.update_position(new_time)

    def forward_to_end(self):
        if self.player.has_data():
            duration = self.player.get_duration()
            self.player.set_time(duration)
            self.visualization.playhead.update_position(duration)
            if self.player.buffer_loaded:
                self.player.stop()  # Buffer mode: stop to reset state, replay starts from beginning
            elif self.player.loaded:
                self.player.reload_file()  # File mode: force reload for replay consistency

    def toggle_record(self):
        if self.audio_manager.source != "mic":
            print("🎙 Starting recording...")
            self.audio_manager.start_mic()
            self.player.stop()
            self.player.unload_buffer()
            self.visualization.set_mode("Waveform")
            self.visualization.start_recording_mode()
            self.record_btn.setText("⏹ Stop")
            self.play_pause_btn.setEnabled(False)
            self.file_display.clear()
        else:
            print("🛑 Stopping recording...")
            self.audio_manager.stop_mic()
            self.visualization.stop_recording_mode()

            recorded = self.audio_manager.get_recording_data()
            if len(recorded) > 0:
                self.player.load_buffer(recorded, self.audio_manager.get_samplerate())
                self.visualization.plot_waveform(recorded, len(recorded) / self.audio_manager.get_samplerate())
                self.play_pause_btn.setEnabled(True)
                self.play_pause_btn.setText("▶️ Play")
            else:
                print("⚠️ No data recorded.")
                self.play_pause_btn.setEnabled(False)
            self.record_btn.setText("🔴 Record")

    def toggle_live_mode(self):
        self.isLive = not self.isLive
        if self.isLive:
            print("✅ Live Feed mode activated")
            self.start_live_mode()
            self.live_btn.setText("Stop Live Mode")
            self.play_pause_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
        else:
            print("🛑 Live Feed mode deactivated")
            self.stop_live_mode()
            self.live_btn.setText("Live Mode")
            self.play_pause_btn.setEnabled(True)
            self.record_btn.setEnabled(True)

    #@Slot(str, bool)
    def update_live_status_label(self, text, is_pitch):
        if is_pitch:
            self.pitch_label.setText(text)
        else:
            self.harmonic_label.setText(text)

    def start_live_mode(self):
        def live_callback(audio_chunk):
            result = self.analysis_engine.process_live_frame(audio_chunk)
            self.visualization.update_live_visualization(result)

        self.audio_manager.start_live_mode(live_callback)
        self.visualization.set_mode(self.visualization.mode)

    def stop_live_mode(self):
        self.audio_manager.stop_current_stream()
        self.visualization.clear_visualization()  # Optional

    def change_mode(self, mode):
        self.visualization.set_mode(mode)

    def run(self):
        self.visualization.start_loop(self.audio_manager, self.player, self.analysis_engine)

    def keyPressEvent(self, event):
        if self.audio_manager.source == "mic":
            if event.key() == Qt.Key_Space:
                self.toggle_record()
                event.accept()
        elif self.player.has_data():
            if event.key() == Qt.Key_Space and self.play_pause_btn.isEnabled():
                self.toggle_play()
                event.accept()
        else:
            super().keyPressEvent(event)
