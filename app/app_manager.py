# app/app_state.py

class AppState:
    def __init__(self):
        self.isLive = False
        self.isRecording = False
        self.isPlaying = False
        self.isEnded = False
        self.hasFile = False
        self.isOscilloscope = False
        self.showGroundTruth = False
        self.pitch_algorithm = "SWIPE"

        # analysis
        self.hps_k = 4  # default hps parameter
        self.max_pitch = 700.0
        self.min_pitch = 20.0

