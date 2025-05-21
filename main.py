import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


from app.audio_manager import AudioManager
from app.player import Player
from app.analysis_engine import AnalysisEngine
from app.gui import AppGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)

    audio_manager = AudioManager()
    player = Player()
    analysis_engine = AnalysisEngine()

    gui = AppGUI(audio_manager, player, analysis_engine)
    gui.show()

    QTimer.singleShot(0, gui.run)

    sys.exit(app.exec())
