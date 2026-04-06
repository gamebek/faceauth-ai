import os
import sys
import time
import pickle
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    import config
except ImportError:
    from .. import config

try:
    from src.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image


def load_model(model_path=config.MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    return model


def capture_face_image(timeout_seconds=15):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Could not access the webcam."

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        ret, frame = cap.read()
        if not ret:
            continue

        processed_face = preprocess_image(image_array=frame)
        if processed_face is not None:
            cap.release()
            return processed_face, None

    cap.release()
    return None, "No face detected. Please hold still and look at the camera."


def predict_face(preprocessed_face, model, threshold=config.DISTANCE_THRESHOLD):
    face_vector = preprocessed_face.flatten().reshape(1, -1)

    if not hasattr(model, 'kneighbors'):
        label = model.predict(face_vector)[0]
        return label, None, f"Access Granted: {label}"

    distances, _ = model.kneighbors(face_vector)
    distance = float(distances[0][0])

    if distance > threshold:
        return None, distance, "Access Denied"

    label = model.predict(face_vector)[0]
    return label, distance, f"Access Granted: {label}"


def login_with_face(model_path=config.MODEL_PATH, threshold=config.DISTANCE_THRESHOLD):
    model = load_model(model_path)
    face, error = capture_face_image()
    if face is None:
        return None, None, error

    return predict_face(face, model, threshold)


if __name__ == "__main__":
    label, distance, message = login_with_face()
    print(message)
    if distance is not None:
        print(f"Distance: {distance:.2f}")
