import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from src.feature_engineering import load_data
except ImportError:
    from feature_engineering import load_data

def train_model(data_dir="data/raw", model_save_path="models/face_model.pkl"):
    """
    Trains a KNN classifier on the extracted facial features and saves it to disk.
    """

    # Load and prepare data using the logic from feature_engineering.py
    
    # Fallback to dataset or data/processed if data/raw doesn't exist
    if not os.path.exists(data_dir):
        if os.path.exists("dataset"):
            data_dir = "dataset"
        elif os.path.exists("data/processed"):
            data_dir = "data/processed"
            
    X, y = load_data(data_dir)
    
    if len(X) == 0:
        print("Error: No data loaded. Cannot train model.")
        print("Please ensure you have captured facial images first (Step 2 & 3).")
        return
        

    print("Performing train/test split...")
    
    # Check if we have enough distinct classes for stratification
    unique_classes = np.unique(y)
    stratify_param = y if len(unique_classes) > 1 else None
    
    if len(X) < 2:
        print("Error: Not enough data points to perform train/test split. Capture more images.")
        return
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    except ValueError as e:
        print(f"Warning during split: {e}")
        print("Falling back to unstratified split due to limited data per class.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    print("Initializing and training KNN model...")
    # Initialize KNN classifier - starting with n_neighbors=3
    # If we have very few samples, adjust n_neighbors
    n_neighbors = min(3, len(X_train))
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn_model.fit(X_train, y_train)
    

    print("Evaluating model performance on test set...")
    
    if len(X_test) > 0:
        y_pred = knn_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        if len(unique_classes) > 1:
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            # zero_division=0 prevents warnings when a class has no predicted samples
            print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("Test set is empty, skipping evaluation.")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"\nSaving model to {model_save_path}...")
    # Save the trained model using pickle
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_model, f)
        
    print("Model saved successfully! Training complete.")

if __name__ == "__main__":
    train_model()
