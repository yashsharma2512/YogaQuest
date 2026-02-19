import cv2
import mediapipe as mp
import os
import csv
from tqdm import tqdm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

DATASET_PATH = "dataset"
OUTPUT_FILE = "pose_data.csv"

# Mapping dataset labels → app labels
pose_map = {
    "tree": "Tree Pose",
    "warrior2": "Warrior II",
    "downdog": "Downward Dog",
    "plank": "Plank"
}

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)

    for split in ["train", "test"]:
        for folder in pose_map:

            folder_path = os.path.join(DATASET_PATH, split, folder)

            if not os.path.exists(folder_path):
                continue

            for img_name in tqdm(os.listdir(folder_path), desc=f"{split}-{folder}"):

                img_path = os.path.join(folder_path, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark

                    row = []
                    for lm in landmarks:
                        row.extend([lm.x, lm.y])

                    row.append(pose_map[folder])
                    writer.writerow(row)

print("✅ CSV dataset created successfully!")
