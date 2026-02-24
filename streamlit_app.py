import streamlit as st
import cv2
import numpy as np
import time

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from engine.pose_detector import PoseDetector
from engine.ml_pose_classifier import classify_pose_ml
from engine.scoring_engine import hybrid_score, detect_fatigue


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="YogaQuest AI",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>🧘 YogaQuest AI – Web Version</h1>",
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.header("Session Controls")

pose_name = st.sidebar.selectbox(
    "Select Pose",
    ["Tree Pose", "Warrior II", "Downward Dog", "Plank"]
)

duration = st.sidebar.selectbox(
    "Duration (seconds)",
    [15, 30, 60]
)

start_session = st.sidebar.button("▶ Start Session")

# ================= SESSION STATE =================
if "scores" not in st.session_state:
    st.session_state.scores = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "session_active" not in st.session_state:
    st.session_state.session_active = False


# ================= START SESSION =================
if start_session:
    st.session_state.scores = []
    st.session_state.start_time = time.time()
    st.session_state.session_active = True


# ================= WEBRTC CONFIG =================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ================= VIDEO PROCESSOR =================
class PoseProcessor(VideoTransformerBase):

    def __init__(self):
        self.detector = PoseDetector()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img, landmarks = self.detector.detect(img)

        ml_pose, ml_conf = classify_pose_ml(landmarks)

        score, feedback, errors, confidence = hybrid_score(
            pose_name,
            landmarks,
            ml_pose,
            ml_conf
        )

        # Save score only if session active
        if st.session_state.session_active:
            st.session_state.scores.append(score)

        # ===== Overlay Info =====
        cv2.putText(img, f"Score: {score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"Conf: {round(confidence, 2)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if feedback:
            cv2.putText(img, feedback[0], (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Highlight incorrect joints
        if landmarks:
            h, w, _ = img.shape
            for idx in errors:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

        return img


# ================= LAYOUT =================
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="yoga-ai",
        video_transformer_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Live Metrics")

    if st.session_state.session_active and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = max(0, duration - elapsed)

        st.metric("⏳ Time Left", f"{remaining}s")

        if remaining <= 0:
            st.session_state.session_active = False
            st.success("Session Complete 🎉")

    if st.session_state.scores:
        current = st.session_state.scores[-1]
        avg = int(np.mean(st.session_state.scores))
        fatigue = detect_fatigue(st.session_state.scores)

        st.metric("⭐ Current Score", current)
        st.metric("📈 Average Score", avg)
        st.metric("💪 Fatigue", "YES ⚠️" if fatigue else "No")

        st.line_chart(st.session_state.scores)