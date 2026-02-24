import math
import numpy as np
from collections import deque

# ===== GLOBAL STABILITY BUFFER =====
ANGLE_BUFFER = deque(maxlen=30)  # ~1 second of frames


# ================= ANGLE =================
def calculate_angle(a, b, c):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return angle


# ================= STABILITY =================
def compute_stability(current_angle):

    ANGLE_BUFFER.append(current_angle)

    if len(ANGLE_BUFFER) < 5:
        return 50  # neutral until enough data

    variance = np.var(ANGLE_BUFFER)

    # Convert variance → stability score
    stability_score = max(0, 100 - variance * 5)

    return min(100, stability_score)
# ++++++++++++++++++FATIGUE+++++++++++++++++++++
def detect_fatigue(score_history):

    if len(score_history) < 40:
        return False

    first_avg = np.mean(score_history[:20])
    last_avg = np.mean(score_history[-20:])

    # If performance drops significantly
    if first_avg - last_avg > 15:
        return True

    return False

# ================= HYBRID ROUTER =================
def hybrid_score(pose_name, landmarks, ml_pose, ml_confidence):

    if not landmarks:
        return 0, ["Body not detected"], [], 0

    # ===== ML CONFIDENCE CHECK =====
    if ml_pose != pose_name or ml_confidence < 0.6:
        return 20, ["Pose not confidently detected"], [], ml_confidence

    # ===== RULE SCORE =====
    rule_score, feedback, errors, key_angle = rule_engine(pose_name, landmarks)

    # ===== STABILITY SCORE =====
    stability_score = compute_stability(key_angle)

    # ===== FINAL SCORE =====
    final_score = (
        0.5 * rule_score +
        0.3 * stability_score +
        0.2 * (ml_confidence * 100)
    )

    return int(final_score), feedback, errors, ml_confidence


# ================= RULE ENGINE =================
def rule_engine(pose_name, landmarks):

    if pose_name == "Tree Pose":
        return tree_rules(landmarks)

    if pose_name == "Warrior II":
        return warrior_rules(landmarks)

    if pose_name == "Plank":
        return plank_rules(landmarks)

    if pose_name == "Downward Dog":
        return downdog_rules(landmarks)

    return 0, ["Unknown pose"], [], 0


# ================= TREE =================
def tree_rules(landmarks):

    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

    left_knee = calculate_angle(
        landmarks[LEFT_HIP],
        landmarks[LEFT_KNEE],
        landmarks[LEFT_ANKLE]
    )

    right_knee = calculate_angle(
        landmarks[RIGHT_HIP],
        landmarks[RIGHT_KNEE],
        landmarks[RIGHT_ANKLE]
    )

    if left_knee > 150 and right_knee > 150:
        return 20, ["Lift one leg"], [LEFT_KNEE, RIGHT_KNEE], 170

    score = 100
    feedback = []
    errors = []

    bent_angle = min(left_knee, right_knee)

    if not (60 < bent_angle < 120):
        score -= 30
        feedback.append("Bend lifted leg more")
        errors.extend([LEFT_KNEE, RIGHT_KNEE])

    return score, feedback, errors, bent_angle


# ================= WARRIOR =================
def warrior_rules(landmarks):

    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28

    left_knee = calculate_angle(
        landmarks[LEFT_HIP],
        landmarks[LEFT_KNEE],
        landmarks[LEFT_ANKLE]
    )

    right_knee = calculate_angle(
        landmarks[RIGHT_HIP],
        landmarks[RIGHT_KNEE],
        landmarks[RIGHT_ANKLE]
    )

    if not (70 < left_knee < 110 or 70 < right_knee < 110):
        return 25, ["Front knee 90°"], [LEFT_KNEE, RIGHT_KNEE], 170

    key_angle = left_knee if 70 < left_knee < 110 else right_knee

    return 100, [], [], key_angle


# ================= PLANK =================
def plank_rules(landmarks):

    SHOULDER, HIP, ANKLE = 12, 24, 28

    body_angle = calculate_angle(
        landmarks[SHOULDER],
        landmarks[HIP],
        landmarks[ANKLE]
    )

    if body_angle < 150:
        return 20, ["Lower hips"], [HIP], body_angle

    return 100, [], [], body_angle


# ================= DOWNDOG =================
def downdog_rules(landmarks):

    SHOULDER, HIP, ANKLE = 12, 24, 28

    hip_angle = calculate_angle(
        landmarks[SHOULDER],
        landmarks[HIP],
        landmarks[ANKLE]
    )

    if hip_angle > 140:
        return 20, ["Raise hips"], [HIP], hip_angle

    return 100, [], [], hip_angle