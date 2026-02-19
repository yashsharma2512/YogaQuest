import cv2
import os
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QSoundEffect

from engine.pose_detector import PoseDetector
from engine.scoring_engine import score_pose
from ui.results_window import ResultsWindow
from utils.session_manager import save_session


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()

        # ================= STYLE =================
        self.setStyleSheet("""
            QWidget { background-color: #0D0D0D; color: white; font-family: Segoe UI; }
            QPushButton { background:#1F1F1F; border-radius:10px; padding:10px; }
            QPushButton:hover { background:#2A2A2A; }
            QComboBox { background:#1F1F1F; padding:6px; border-radius:8px; }
        """)

        # ================= STATE =================
        self.cap = None
        self.running = False
        self.session_scores = []

        # ================= TIMERS =================
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)

        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_countdown)

        # ================= SOUND =================
        self.sound = QSoundEffect()
        sound_path = os.path.abspath("assets/success.wav")
        self.sound.setSource(QUrl.fromLocalFile(sound_path))
        self.sound.setVolume(1.0)

        # ================= CAMERA =================
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 400)
        self.camera_label.setStyleSheet("background:black;border-radius:12px;")

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

        self.back_btn = QPushButton("← Home")
        self.back_btn.clicked.connect(self.go_home)

        # ================= LABELS =================
        self.score_label = QLabel("⭐ 0")
        self.score_label.setStyleSheet("font-size: 20px;")

        self.timer_label = QLabel("⏳ 0s")
        self.timer_label.setStyleSheet("font-size: 16px;")

        self.feedback_label = QLabel("🎯 Feedback")
        self.feedback_label.setWordWrap(True)

        # ================= LAYOUT =================
        right = QVBoxLayout()
        right.addWidget(self.back_btn)
        right.addWidget(QLabel("Pose"))
        right.addWidget(self.pose_selector)
        right.addWidget(QLabel("Duration"))
        right.addWidget(self.duration_selector)
        right.addWidget(self.start_btn)
        right.addWidget(self.stop_btn)
        right.addSpacing(10)
        right.addWidget(self.score_label)
        right.addWidget(self.timer_label)
        right.addWidget(self.feedback_label)
        right.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.camera_label)
        main.addLayout(right)

        self.setLayout(main)

        # ================= ENGINE =================
        self.pose_detector = PoseDetector()
        self.back_to_home = None

    # ================= START =================
    def start_session(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture('planks.mp4')

        self.session_scores = []
        self.remaining_time = int(self.duration_selector.currentText())
        self.running = True

        self.frame_timer.start(30)
        self.session_timer.start(1000)

    # ================= STOP =================
    def stop_session(self):
        if self.running:
            self.end_session()

    # ================= BACK =================
    def go_home(self):
        self.stop_session()
        if self.back_to_home:
            self.back_to_home()

    # ================= TIMER =================
    def update_countdown(self):
        self.remaining_time -= 1
        self.timer_label.setText(f"⏳ {self.remaining_time}s")

        if self.remaining_time <= 0:
            self.end_session()

    # ================= FRAME =================
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame, landmarks = self.pose_detector.detect(frame)

        pose = self.pose_selector.currentText()
        score, feedback, errors = score_pose(pose, landmarks)

        self.session_scores.append(score)

        # ===== UI UPDATE =====
        self.score_label.setText(f"⭐ {score}")

        if feedback:
            self.feedback_label.setText("🎯 " + ", ".join(feedback))
        else:
            self.feedback_label.setText("🎯 Perfect!")

        # ===== JOINT HIGHLIGHT =====
        if landmarks:
            h, w, _ = frame.shape

            # Red = incorrect joints
            for idx in errors:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        # ===== FIXED CAMERA DISPLAY (NO CROPPING) =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        scale = min(640 / w, 400 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(rgb, (new_w, new_h))

        qt_img = QImage(resized.data, new_w, new_h, ch * new_w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # ================= END =================
    def end_session(self):
        self.frame_timer.stop()
        self.session_timer.stop()
        self.running = False

        if self.cap:
            self.cap.release()

        # ===== SAVE SESSION =====
        if self.session_scores:
            save_session(self.session_scores)

        # ===== SOUND =====
        self.sound.play()

        # ===== RESULTS =====
        self.results = ResultsWindow(self.session_scores)
        self.results.show()
