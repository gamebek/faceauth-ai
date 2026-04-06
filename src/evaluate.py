import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from src.feature_engineering import load_data
except ImportError:
    from feature_engineering import load_data

try:
    import config
except ImportError:
    from .. import config


def evaluate_model(model_path=config.MODEL_PATH):
    X, y = load_data()
    if len(X) == 0:
        print("No data found for evaluation. Please capture face images first.")
        return

    unique_classes = np.unique(y)
    if len(unique_classes) == 1:
        print("Warning: Only one registered user found. Evaluation will not reflect impostor rejection.")

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Train the model first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    evaluate_model()
