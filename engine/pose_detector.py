import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.missing_frames = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        landmarks = None

        if result.pose_landmarks:
            self.missing_frames = 0
            self.mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            landmarks = result.pose_landmarks.landmark
        else:
            self.missing_frames += 1

        if self.missing_frames > 30:
            self.pose.close()
            self.pose = self.mp_pose.Pose()
            self.missing_frames = 0

        return frame, landmarks
