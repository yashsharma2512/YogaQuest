from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QLinearGradient, QPainter, QColor
import numpy as np
import pyqtgraph as pg

from utils.session_manager import load_history


class ResultsWindow(QWidget):
    def __init__(self, scores):
        super().__init__()

        self.setMinimumSize(800, 700)

        self.scores = scores
        self.smoothed = self.ema(scores)

        history = load_history()

        avg = int(np.mean(scores))

        layout = QVBoxLayout()

        title = QLabel("🏁 Session Complete")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_score = QLabel(f"{avg}")
        main_score.setFont(QFont("Segoe UI", 64, QFont.Weight.Bold))
        main_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_score.setStyleSheet("color:#6C63FF;")

        layout.addWidget(title)
        layout.addWidget(main_score)

        # ===== GRAPH =====
        self.graph = pg.PlotWidget()
        self.graph.setBackground("#111")

        self.curve = self.graph.plot(
            [], pen=pg.mkPen(color=(108, 99, 255), width=3)
        )

        # Compare with previous session (smoothed)
        if len(history) > 1:
            prev = history[-2]
            prev_smooth = self.ema(prev)

            self.graph.plot(
                prev_smooth,
                pen=pg.mkPen(color=(200, 200, 200), width=2, style=Qt.PenStyle.DashLine)
            )

        layout.addWidget(self.graph)

        # ===== AI FEEDBACK =====
        feedback = QLabel("🧠 Trend shows improving stability 📈")
        feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(feedback)

        btn = QPushButton("Back")
        btn.clicked.connect(self.close)

        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

        # Animate graph
        self.index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    # ===== EMA SMOOTHING =====
    def ema(self, data, alpha=0.2):
        ema = []
        value = data[0]

        for d in data:
            value = alpha * d + (1 - alpha) * value
            ema.append(value)

        return ema

    # ===== ANIMATION =====
    def animate(self):
        if self.index >= len(self.smoothed):
            self.timer.stop()
            return

        self.curve.setData(self.smoothed[:self.index])
        self.index += 1

    # ===== BACKGROUND =====
    def paintEvent(self, event):
        painter = QPainter(self)

        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#141E30"))
        gradient.setColorAt(1, QColor("#243B55"))

        painter.fillRect(self.rect(), gradient)
