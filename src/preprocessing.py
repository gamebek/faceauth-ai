import cv2
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Load Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def preprocess_image(image_path=None, image_array=None):
    """
    Reads an image (from path or array), detects the face, crops it, 
    resizes to config.IMAGE_SIZE, converts to grayscale, and normalizes it.
    Returns the preprocessed face image numpy array or None if no face is detected.
    """
    if image_path is not None:
        img = cv2.imread(image_path)
    elif image_array is not None:
        img = image_array
    else:
        raise ValueError("Either image_path or image_array must be provided.")
        
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Step 1: Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Detect face using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None # No face detected
        
    # For preprocessing, we simplify by just taking the first face detected
    x, y, w, h = faces[0]
    
    # Step 3: Crop face region
    face_roi = gray[y:y+h, x:x+w]
    
    # Step 4: Resize image
    face_resized = cv2.resize(face_roi, config.IMAGE_SIZE)
    
    # Step 5: Normalize pixel values (scaling pixels to range [0, 1])
    face_normalized = face_resized / 255.0
    
    return face_normalized

def test_preprocessing():
    """Simple test function to verify preprocessing works on a captured image."""
    print("Testing preprocessing logic...")
    raw_dirs = os.listdir(config.RAW_DATA_DIR) if os.path.exists(config.RAW_DATA_DIR) else []
    
    if not raw_dirs:
        print("Please capture some images first using data_collection.py to test preprocessing.")
        return
        
    user_test_dir = os.path.join(config.RAW_DATA_DIR, raw_dirs[0])
    images = os.listdir(user_test_dir)
    
    if not images:
        print(f"No images found in {user_test_dir}")
        return
        
    test_image_path = os.path.join(user_test_dir, images[0])
    print(f"Processing: {test_image_path}")
    
    processed_face = preprocess_image(image_path=test_image_path)
    if processed_face is not None:
        print(f"✅ Success! Preprocessed shape: {processed_face.shape}")
        # Note: image is normalized (0-1), so display might be dark without scaling back to 0-255
        cv2.imshow("Preprocessed Face", processed_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No face detected in the image.")

if __name__ == "__main__":
    test_preprocessing()
