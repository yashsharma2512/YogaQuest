import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from engine.pose_detector import PoseDetector
from engine.scoring_engine import score_pose


# ================= PAGE CONFIG =================
st.set_page_config(page_title="YogaQuest AI", layout="wide")

# ================= STYLE =================
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }
    .big-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧘 YogaQuest AI</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Controls")

pose_name = st.sidebar.selectbox("Select Pose", [
    "Tree Pose",
    "Warrior II",
    "Downward Dog",
    "Plank"
])

duration = st.sidebar.selectbox("Session Duration", [15, 30, 60])

start = st.sidebar.button("▶ Start Session")

# ================= STATE =================
if "scores" not in st.session_state:
    st.session_state.scores = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None


# ================= RTC CONFIG =================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ================= VIDEO PROCESSOR =================
class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = PoseDetector()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img, landmarks = self.detector.detect(img)

        score, feedback, errors = score_pose(pose_name, landmarks)

        # Save score
        st.session_state.scores.append(score)

        # Overlay text
        cv2.putText(img, f"Score: {score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if feedback:
            cv2.putText(img, feedback[0], (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Highlight errors
        if landmarks:
            h, w, _ = img.shape
            for idx in errors:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

        return img


# ================= START SESSION =================
if start:
    st.session_state.scores = []
    st.session_state.start_time = time.time()


# ================= MAIN LAYOUT =================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera")

    webrtc_streamer(
        key="yoga",
        video_transformer_factory=PoseTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


with col2:
    st.subheader("📊 Live Analytics")

    # ===== TIMER =====
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = max(0, duration - elapsed)

        st.metric("⏳ Time Left", f"{remaining}s")

        if remaining <= 0:
            st.success("Session Complete 🎉")

    # ===== SCORE =====
    if st.session_state.scores:
        current = st.session_state.scores[-1]
        avg = int(np.mean(st.session_state.scores))

        st.metric("⭐ Current Score", current)
        st.metric("📈 Average Score", avg)

        # ===== GRAPH =====
        st.line_chart(st.session_state.scores)

        # ===== STABILITY =====
        stability = int(np.std(st.session_state.scores))
        st.metric("🎯 Stability", stability)

        # ===== AI COACH =====
        st.subheader("🧠 AI Coach")

        suggestions = []

        if avg < 60:
            suggestions.append("Focus on posture basics")

        if stability > 15:
            suggestions.append("Try holding the pose steadily")

        if avg > 80:
            suggestions.append("Great form! Increase duration")

        if not suggestions:
            suggestions.append("Maintain consistency")

        for s in suggestions:
            st.write("•", s)
