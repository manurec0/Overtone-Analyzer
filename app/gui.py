from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                               QLineEdit, QLabel, QCheckBox, QDialog, QComboBox)
from PySide6.QtCore import Qt
from app.visualization import Visualization


class AppGUI(QMainWindow):
    def __init__(self, audio_manager, player, analysis_engine, app_state):
        super().__init__()

        self.app_state = app_state
        self.audio_manager = audio_manager
        self.player = player
        self.analysis_engine = analysis_engine
        self.pitch_algorithm = "SWIPE"

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
        self.rewind_5s_btn = QPushButton("⏪", clicked=self.rewind_5s)
        self.play_pause_btn = QPushButton("▶️ Play", clicked=self.toggle_play)
        self.play_pause_btn.setEnabled(False)
        self.forward_5s_btn = QPushButton("⏩", clicked=self.forward_5s)
        self.forward_end_btn = QPushButton("⏭", clicked=self.forward_to_end)
        self.record_btn = QPushButton("🔴", clicked=self.toggle_record)
        self.settings_btn = QPushButton("Settings", clicked=self.open_audio_settings)
        self.live_btn = QPushButton("Real-Time Mode", clicked=self.toggle_live_mode)

        self.oscilloscope_btn = QCheckBox("Oscilloscope Mode")
        self.oscilloscope_btn.setEnabled(False)
        self.oscilloscope_btn.clicked.connect(self.toggle_oscilloscope)

        controls_inner_layout.addWidget(self.rewind_start_btn)
        controls_inner_layout.addWidget(self.rewind_5s_btn)
        controls_inner_layout.addWidget(self.play_pause_btn)
        controls_inner_layout.addWidget(self.forward_5s_btn)
        controls_inner_layout.addWidget(self.forward_end_btn)
        controls_inner_layout.addWidget(self.record_btn)

        top_bar_layout.addWidget(self.settings_btn)
        top_bar_layout.addWidget(self.live_btn)
        top_bar_layout.addWidget(self.oscilloscope_btn)

        top_bar_layout.addStretch()
        top_bar_layout.addLayout(controls_inner_layout)
        top_bar_layout.addStretch()

        # Pitch Detection Selector (far right)
        self.pitch_selector = QComboBox()
        self.pitch_selector.addItems(["SWIPE", "YIN", "pYIN", "HPS"])
        self.pitch_selector.setCurrentText("SWIPE")  # default
        self.pitch_selector.setToolTip("Select pitch detection algorithm")
        self.pitch_selector.currentTextChanged.connect(self.change_pitch_algorithm)
        top_bar_layout.addWidget(QLabel("Pitch Detection Method:"))
        top_bar_layout.addWidget(self.pitch_selector)

        main_layout.addLayout(top_bar_layout)

        # Ground Truth + Pitch Info Bar Below Top Bar
        info_bar_layout = QHBoxLayout()
        info_bar_layout.setContentsMargins(10, 0, 10, 0)  # Optional spacing

        # Left side: Pitch and Harmonic labels
        info_labels_layout = QVBoxLayout()
        info_labels_layout.setAlignment(Qt.AlignLeft)

        self.pitch_label = QLabel("Note: ...")
        self.harmonic_label = QLabel("Active Harmonic: ...")
        self.pitch_label.setStyleSheet("color: white; font-size: 14px;")
        self.harmonic_label.setStyleSheet("color: white; font-size: 14px;")

        info_labels_layout.addWidget(self.pitch_label)
        info_labels_layout.addWidget(self.harmonic_label)

        info_bar_layout.addLayout(info_labels_layout)

        info_bar_layout.addStretch()  # Push ground truth toggle to the right

        self.ground_truth_checkbox = QCheckBox("Overlay Ground Truth")
        self.ground_truth_checkbox.setEnabled(False)
        self.ground_truth_checkbox.clicked.connect(self.toggle_ground_truth)
        info_bar_layout.addWidget(self.ground_truth_checkbox)

        # Add to main layout just below top bar
        main_layout.addLayout(info_bar_layout)

        # Visualization below labels
        self.visualization = Visualization(app_state)
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

        self.clear_file_btn = QPushButton("❌ Clear", clicked=self.clear_loaded_file)
        bottom_layout.addWidget(self.load_btn)
        bottom_layout.addWidget(self.file_display)

        bottom_layout.addWidget(self.load_btn)
        bottom_layout.addWidget(self.file_display)
        bottom_layout.addWidget(self.clear_file_btn)

        bottom_layout.addStretch()

        # Mode Drop Down Menu
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Waveform",
            "Fundamental Pitch Detection",
            "Spectrogram",
            "Overtone Profile",
            "Overtone Analyzer"])
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        bottom_layout.addWidget(self.mode_selector)

    def load_wav(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV files (*.wav)")
        if filepath:
            duration = self.audio_manager.load_wav(filepath)
            self.player.load_buffer(self.audio_manager.wav_data, self.audio_manager.get_samplerate())
            self.visualization.set_mode(self.visualization.mode)  # Automatically plots depending on mode
            self.file_display.setText(filepath)
            self.play_pause_btn.setEnabled(True)
            self.play_pause_btn.setText("▶️ Play")
            self.oscilloscope_btn.setEnabled(True)

    def toggle_play(self):
        if self.player.has_data():
            if self.player.is_playing():
                self.player.pause()
                self.app_state.isPlaying = False
                self.play_pause_btn.setText("▶️ Play")
            else:
                self.player.play()
                self.app_state.isPlaying = True
                self.play_pause_btn.setText("⏸ Pause")

    def rewind_to_start(self):
        if self.player.has_data():
            was_playing = self.player.is_playing()
            if was_playing:
                self.player.pause()

            self.player.set_time(0)
            self.visualization.playhead.update_position(0)

            if was_playing:
                self.player.play()

    def rewind_5s(self):
        if self.player.has_data():
            was_playing = self.player.is_playing()
            if was_playing:
                self.player.pause()

            current = self.player.get_time()
            new_time = max(0, current - 5)
            self.player.set_time(new_time)
            self.visualization.playhead.update_position(new_time)

            if was_playing:
                self.player.play()

    def forward_5s(self):
        if self.player.has_data():
            was_playing = self.player.is_playing()
            if was_playing:
                self.player.pause()

            current = self.player.get_time()
            new_time = min(self.player.get_duration(), current + 5)
            self.player.set_time(new_time)
            self.visualization.playhead.update_position(new_time)

            if was_playing:
                self.player.play()

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
        self.app_state.isLive = not self.app_state.isLive
        if self.app_state.isLive:
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

    def toggle_oscilloscope(self):
        self.app_state.isOscilloscope = self.oscilloscope_btn.isChecked()

        # Update X-axis label based on oscilloscope state
        if self.app_state.isOscilloscope and self.visualization.mode == "Waveform" and not self.app_state.isLive:
            self.visualization.plot_item.setLabel('bottom', 'Samples')
        elif not self.app_state.isOscilloscope and self.visualization.mode == "Waveform" and not self.app_state.isLive:
            self.visualization.plot_item.setLabel('bottom', 'Time (s)')
            # Also restore waveform display if toggled off
            self.visualization.set_mode("Waveform")

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
        # Safety pause to avoid lag during heavy visualizations
        if self.player.is_playing():
            print("⏸ Auto-pausing due to mode change")
            self.player.pause()
            self.app_state.isPlaying = False
            self.play_pause_btn.setText("▶️ Play")

        self.visualization.set_mode(mode)

        if self.player.has_data():
            current_time = self.player.get_time()
            self.visualization.playhead.update_position(current_time)

        # Oscilloscope toggle only enabled in Waveform mode and when a file is loaded
        if mode == "Waveform" and self.player.buffer_loaded:
            self.oscilloscope_btn.setEnabled(True)
        else:
            self.oscilloscope_btn.setChecked(False)
            self.oscilloscope_btn.setEnabled(False)
            self.app_state.isOscilloscope = False

        # Ground truth toggle only enabled in specific modes and when file is loaded
        if mode in ["Fundamental Pitch Detection", "Overtone Analyzer"] and self.player.has_data():
            self.ground_truth_checkbox.setEnabled(True)
        else:
            self.ground_truth_checkbox.setChecked(False)
            self.ground_truth_checkbox.setEnabled(False)
            self.app_state.showGroundTruth = False

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

    def clear_loaded_file(self):
        self.player.stop()

        # Reset file-related state
        self.audio_manager.unload_wav()
        self.player.unload_buffer()

        # Disable playback controls
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setText("▶️ Play")

        # Clear visualization and GUI
        self.visualization.clear_visualization()
        self.visualization.clear_caches()
        self.visualization.playhead.update_position(0)
        self.file_display.clear()

        self.oscilloscope_btn.setChecked(False)  # Reset checkbox
        self.oscilloscope_btn.setEnabled(False)  # Disable checkbox

        self.ground_truth_checkbox.setChecked(False)
        self.ground_truth_checkbox.setEnabled(False)
        self.app_state.showGroundTruth = False

        print("🧹 Cleared loaded WAV file and reset visualizations.")

    def open_audio_settings(self):
        dialog = AudioSettingsDialog(self.audio_manager, self)
        dialog.exec()

    def change_pitch_algorithm(self, algorithm):
        self.app_state.pitch_algorithm = algorithm
        print(f"🎚 Pitch algorithm set to: {algorithm}")
        print("♻️ Clearing pitch cache due to algorithm change")
        self.visualization.clear_caches()

        if self.visualization.mode == "Fundamental Pitch Detection":
            self.visualization.set_mode("Fundamental Pitch Detection")  # re-trigger replot
        if self.visualization.mode == "Overtone Analyzer":
            self.visualization.set_mode("Overtone Analyzer")

    def toggle_ground_truth(self):
        self.app_state.showGroundTruth = self.ground_truth_checkbox.isChecked()
        print(f"📊 Ground Truth Overlay: {'ON' if self.app_state.showGroundTruth else 'OFF'}")
        self.visualization.toggle_ground_truth_overlay(self.app_state.showGroundTruth)


class AudioSettingsDialog(QDialog):
    def __init__(self, audio_manager, parent=None):
        super().__init__(parent)
        self.audio_manager = audio_manager
        self.setWindowTitle("Audio Input Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()
        self.device_selector = QComboBox()
        self.devices = self.audio_manager.list_input_devices()

        for i, name in self.devices:
            self.device_selector.addItem(f"{i}: {name}", i)

        layout.addWidget(QLabel("Select Microphone Input:"))
        layout.addWidget(self.device_selector)

        self.confirm_btn = QPushButton("Apply")
        self.confirm_btn.clicked.connect(self.apply_selection)
        layout.addWidget(self.confirm_btn)

        self.setLayout(layout)

    def apply_selection(self):
        selected_index = self.device_selector.currentData()
        print(f"🔧 Selected device index: {selected_index}")
        self.audio_manager.set_input_device(selected_index)
        self.accept()
