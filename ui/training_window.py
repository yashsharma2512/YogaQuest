import cv2
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QComboBox, QPushButton,
    QFileDialog, QFrame
)
from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtMultimedia import QSoundEffect

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from engine.pose_detector import PoseDetector
from engine.ml_pose_classifier import classify_pose_ml
from engine.scoring_engine import hybrid_score, detect_fatigue
from ui.results_window import ResultsWindow


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YogaQuest AI")

        # ================= STYLE =================
        self.setStyleSheet("""
            QWidget {
                background-color: #0F172A;
                color: white;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #1E293B;
                border-radius: 10px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            QComboBox {
                background-color: #1E293B;
                padding: 6px;
                border-radius: 6px;
            }
        """)

        # ================= STATE =================
        self.cap = None
        self.running = False
        self.session_scores = []
        self.session_summary = {}

        # ================= TIMERS =================
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)

        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_countdown)

        # ================= SOUND =================
        self.sound = QSoundEffect()
        self.sound.setSource(QUrl.fromLocalFile(
            os.path.abspath("assets/success.wav")
        ))

        # ================= CAMERA =================
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 400)
        self.camera_label.setStyleSheet(
            "background-color:black;border-radius:15px;"
        )

        # ================= CONTROLS =================
        self.pose_selector = QComboBox()
        self.pose_selector.addItems([
            "Tree Pose",
            "Warrior II",
            "Downward Dog",
            "Plank"
        ])

        self.duration_selector = QComboBox()
        self.duration_selector.addItems(["15", "30", "60"])

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_session)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_session)

        self.export_btn = QPushButton("📄 Export PDF")
        self.export_btn.clicked.connect(self.export_pdf)

        # ================= METRICS =================
        self.score_label = QLabel("Score: 0")
        self.score_label.setFont(QFont("Segoe UI", 18))

        self.timer_label = QLabel("Time: 0")
        self.fatigue_label = QLabel("Fatigue: No")
        self.feedback_label = QLabel("Feedback")
        self.feedback_label.setWordWrap(True)

        # ================= RIGHT CARD =================
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border-radius: 15px;
                padding: 12px;
            }
        """)

        right_layout = QVBoxLayout(card)
        right_layout.addWidget(QLabel("Pose"))
        right_layout.addWidget(self.pose_selector)
        right_layout.addWidget(QLabel("Duration"))
        right_layout.addWidget(self.duration_selector)
        right_layout.addWidget(self.start_btn)
        right_layout.addWidget(self.stop_btn)
        right_layout.addWidget(self.export_btn)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.score_label)
        right_layout.addWidget(self.timer_label)
        right_layout.addWidget(self.fatigue_label)
        right_layout.addWidget(self.feedback_label)
        right_layout.addStretch()

        # ================= MAIN LAYOUT =================
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label, 3)
        main_layout.addWidget(card, 1)

        self.setLayout(main_layout)

        self.pose_detector = PoseDetector()

    # ================= START =================
    def start_session(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video file
        self.session_scores = []
        self.remaining_time = int(self.duration_selector.currentText())
        self.running = True

        self.frame_timer.start(30)
        self.session_timer.start(1000)

    # ================= STOP =================
    def stop_session(self):
        if self.running:
            self.end_session()

    # ================= TIMER =================
    def update_countdown(self):
        self.remaining_time -= 1
        self.timer_label.setText(f"Time: {self.remaining_time}s")

        if self.remaining_time <= 0:
            self.end_session()

    # ================= FRAME UPDATE =================
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame, landmarks = self.pose_detector.detect(frame)

        pose = self.pose_selector.currentText()
        ml_pose, ml_conf = classify_pose_ml(landmarks)

        score, feedback, errors, conf = hybrid_score(
            pose, landmarks, ml_pose, ml_conf
        )

        self.session_scores.append(score)

        fatigue = detect_fatigue(self.session_scores)

        self.score_label.setText(f"Score: {score}")
        self.feedback_label.setText(
            ", ".join(feedback) if feedback else "Perfect"
        )
        self.fatigue_label.setText(
            "Fatigue: YES ⚠️" if fatigue else "Fatigue: No"
        )

        # ===== DISPLAY FRAME PROPERLY SCALED =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        scale = min(640 / w, 400 / h)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        qt_img = QImage(
            resized.data,
            resized.shape[1],
            resized.shape[0],
            resized.shape[2] * resized.shape[1],
            QImage.Format.Format_RGB888
        )

        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # ================= END SESSION =================
    def end_session(self):

        if not self.running:
            return

        self.running = False

        self.frame_timer.stop()
        self.session_timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        if not self.session_scores:
            return

        self.session_summary = {
            "date": str(datetime.datetime.now()),
            "pose": self.pose_selector.currentText(),
            "average": float(np.mean(self.session_scores)),
            "max": max(self.session_scores),
            "min": min(self.session_scores),
            "fatigue": detect_fatigue(self.session_scores)
        }

        self.sound.play()

        self.results_window = ResultsWindow(self.session_scores)
        self.results_window.show()

    # ================= PDF EXPORT =================
    def export_pdf(self):

        if not self.session_scores:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "Yoga_Report.pdf",
            "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        graph_path = "temp_graph.png"
        plt.figure()
        plt.plot(self.session_scores)
        plt.title("Session Performance")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.savefig(graph_path)
        plt.close()

        doc = SimpleDocTemplate(file_path)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("YogaQuest AI Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        for key, value in self.session_summary.items():
            elements.append(
                Paragraph(f"<b>{key}:</b> {value}", styles["Normal"])
            )
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Image(graph_path, width=5 * inch, height=3 * inch))

        doc.build(elements)

        os.remove(graph_path)