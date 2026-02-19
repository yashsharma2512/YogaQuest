import sys
from PyQt6.QtWidgets import QApplication, QStackedWidget

from ui.home_screen import HomeScreen
from ui.training_window import TrainingWindow


class AppController(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.home = HomeScreen()
        self.training = TrainingWindow()

        self.addWidget(self.home)
        self.addWidget(self.training)

        self.setCurrentWidget(self.home)

        self.home.start_clicked.connect(self.start_training)
        self.training.back_to_home = self.go_home

    def start_training(self):
        self.setCurrentWidget(self.training)

    def go_home(self):
        self.setCurrentWidget(self.home)


app = QApplication(sys.argv)
window = AppController()
window.setWindowTitle("YogaQuest AI")
window.resize(1000, 700)
window.show()
sys.exit(app.exec())
