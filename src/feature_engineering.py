import os
import sys
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
    from src.preprocessing import get_face_features
except ImportError:
    from preprocessing import get_face_features


def load_data(data_dir=None):
    """
    Loads face images from the given directory, extracts face features, and creates the X and y datasets.
    """
    if data_dir is None:
        data_dir = config.RAW_DATA_DIR

    if not os.path.exists(data_dir):
        if os.path.exists(config.DATASET_DIR):
            data_dir = config.DATASET_DIR
        elif os.path.exists(config.PROCESSED_DATA_DIR):
            data_dir = config.PROCESSED_DATA_DIR
        else:
            print(f"Directory {data_dir} does not exist. Please check your data path.")
            return np.array([]), np.array([])

    X = []
    y = []

    print(f"Loading images from {data_dir}...")

    for user_name in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_name)
        if not os.path.isdir(user_path):
            continue

        print(f"Processing images for user: {user_name}")

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            if not os.path.isfile(img_path):
                continue

            processed_face = get_face_features(image_path=img_path)
            if processed_face is None:
                print(f"Warning: No face detected in {img_path}. Skipping.")
                continue

            X.append(processed_face)
            y.append(user_name)

    X = np.array(X)
    y = np.array(y)

    print(f"Successfully loaded {len(X)} images corresponding to {len(np.unique(y)) if len(y) > 0 else 0} users.")
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    if len(X) > 0:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
