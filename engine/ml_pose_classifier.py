import joblib
import numpy as np

model = joblib.load("pose_model.pkl")


def classify_pose_ml(landmarks):
    if landmarks is None:
        return "No Pose", 0

    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y])

    try:
        probs = model.predict_proba([row])[0]
        classes = model.classes_

        idx = np.argmax(probs)

        pose = classes[idx]
        confidence = probs[idx]

        return pose, confidence

    except:
        return "Unknown", 0
