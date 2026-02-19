from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt6.QtCore import pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QLinearGradient, QPainter


class HomeScreen(QWidget):

    start_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        title = QLabel("🧘 YogaQuest AI")
        title.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Train smarter • Move better • Feel stronger")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #BBBBBB;")

        start_btn = QPushButton("Start Training")
        start_btn.setFixedWidth(250)

        start_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(108, 99, 255, 0.9);
                border-radius: 20px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
                color: white;
            }
        """)

        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(30)
        glow.setColor(QColor(108, 99, 255))
        glow.setOffset(0)
        start_btn.setGraphicsEffect(glow)

        start_btn.clicked.connect(self.animate_and_start)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(40)
        layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)

    def animate_and_start(self):
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(400)
        self.anim.setStartValue(1)
        self.anim.setEndValue(0.3)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.anim.finished.connect(self.start_clicked.emit)
        self.anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)

        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#0F2027"))
        gradient.setColorAt(1, QColor("#2C5364"))

        painter.fillRect(self.rect(), gradient)
