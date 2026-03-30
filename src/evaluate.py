import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/raw"

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img.flatten()

X = []
y = []

# Load images
for user in os.listdir(DATASET_PATH):
    user_path = os.path.join(DATASET_PATH, user)

    for img_name in os.listdir(user_path):
        img_path = os.path.join(user_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = preprocess(img)
        X.append(processed)
        y.append(user)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Load model
with open("models/face_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)