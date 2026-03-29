import os
import cv2
import numpy as np

def load_data(data_dir="data/raw"):
    """
    Loads preprocessed images from the given directory, flattens them into 1D arrays,
    and creates the X (features) and y (labels) datasets.
    
    Assumes the directory structure is:
    data_dir/
        user1/
            img1.jpg
            img2.jpg
        user2/
            img1.jpg
            ...
    """
    X = []
    y = []
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        # Fallback to dataset if data/raw doesn't exist
        if os.path.exists("dataset"):
            data_dir = "dataset"
        elif os.path.exists("data/processed"):
            data_dir = "data/processed"
        else:
            print(f"Directory {data_dir} does not exist. Please check your data path.")
            return np.array(X), np.array(y)

    print(f"Loading images from {data_dir}...")
    
    # Iterate through each user folder
    for user_name in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_name)
        
        # Only process directories
        if not os.path.isdir(user_path):
            continue
            
        print(f"Processing images for user: {user_name}")
        
        # Read each image for the user
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            
            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Flatten the image into a 1D array
            # If image is 64x64, flatten makes it 4096-dimensional
            flattened_img = img.flatten()
            
            X.append(flattened_img)
            y.append(user_name)
            
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Successfully loaded {len(X)} images corresponding to {len(np.unique(y)) if len(y) > 0 else 0} users.")
    return X, y

if __name__ == "__main__":
    # Test the function if run directly
    X, y = load_data()
    if len(X) > 0:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
