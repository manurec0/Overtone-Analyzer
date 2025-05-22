# app/app_state.py

class AppState:
    def __init__(self):
        self.isLive = False
        self.isRecording = False
        self.isPlaying = False
        self.isEnded = False
        self.hasFile = False
        self.isOscilloscope = False
