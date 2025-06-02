import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


from app.audio_manager import AudioManager
from app.player import Player
from app.analysis_engine import AnalysisEngine
from app.gui import AppGUI
from app.app_manager import AppState

if __name__ == "__main__":
    app = QApplication(sys.argv)

    audio_manager = AudioManager()
    player = Player()
    app_state = AppState()
    analysis_engine = AnalysisEngine(app_state)

    gui = AppGUI(audio_manager, player, analysis_engine, app_state)
    gui.show()

    QTimer.singleShot(0, gui.run)

    sys.exit(app.exec())
