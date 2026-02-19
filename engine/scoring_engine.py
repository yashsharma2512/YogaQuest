import numpy as np

REQUIRED_LANDMARKS = [11,12,23,24,25,26,27,28]


def full_body_visible(landmarks, threshold=0.6):
    for idx in REQUIRED_LANDMARKS:
        if landmarks[idx].visibility < threshold:
            return False
    return True


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def score_tree_pose(landmarks):
    score = 0
    feedback = []
    errors = []

    def lm(i): return [landmarks[i].x, landmarks[i].y]

    knee = calculate_angle(lm(23), lm(25), lm(27))
    if knee > 165:
        score += 80
    else:
        feedback.append("Straighten leg")
        errors.append(25)

    return score, feedback, errors


def score_warrior_ii(landmarks):
    score = 0
    feedback = []
    errors = []

    def lm(i): return [landmarks[i].x, landmarks[i].y]

    knee = calculate_angle(lm(23), lm(25), lm(27))
    if 70 < knee < 110:
        score += 80
    else:
        feedback.append("Bend knee properly")
        errors.append(25)

    return score, feedback, errors


def score_downdog(landmarks):
    score = 0
    feedback = []
    errors = []

    def lm(i): return [landmarks[i].x, landmarks[i].y]

    hip = calculate_angle(lm(11), lm(23), lm(27))
    if hip < 90:
        score += 80
    else:
        feedback.append("Lift hips higher")
        errors.append(23)

    return score, feedback, errors


def score_plank(landmarks):
    score = 0
    feedback = []
    errors = []

    def lm(i): return [landmarks[i].x, landmarks[i].y]

    shoulder = lm(11)
    hip = lm(23)
    ankle = lm(27)

    angle = calculate_angle(shoulder, hip, ankle)

    if angle > 165:
        score += 80
    else:
        feedback.append("Keep body straight")
        errors.append(23)

    return score, feedback, errors


def score_pose(pose_name, landmarks):

    if landmarks is None:
        return 0, ["No person detected"], []

    if not full_body_visible(landmarks):
        return 0, ["Step back"], []

    if pose_name == "Tree Pose":
        return score_tree_pose(landmarks)

    if pose_name == "Warrior II":
        return score_warrior_ii(landmarks)

    if pose_name == "Downward Dog":
        return score_downdog(landmarks)

    if pose_name == "Plank":
        return score_plank(landmarks)

    return 0, ["Pose unclear"], []
