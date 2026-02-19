import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("pose_data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset size:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("🎯 Accuracy:", accuracy)

joblib.dump(model, "pose_model.pkl")

print("✅ Model saved!")
