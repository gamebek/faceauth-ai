import cv2
import os
import time
import sys

# Add the project root to sys.path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def capture_user_images(user_name, num_images=None):
    """
    Captures images from the webcam and saves them for the specified user.
    """
    if num_images is None:
        num_images = config.NUM_IMAGES_TO_CAPTURE

    user_dir = os.path.join(config.RAW_DATA_DIR, user_name)
    os.makedirs(user_dir, exist_ok=True)

    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print(f"Starting image capture for user: {user_name}")
    print(f"Please look at the camera. Capturing {num_images} images...")

    # Wait for the camera to warm up
    time.sleep(2)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        cv2.imshow("Registration - Capturing Faces", frame)

        # Save the captured image into the user directory
        img_path = os.path.join(user_dir, f"{user_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Captured {count+1}/{num_images}: {img_path}")

        count += 1

        # Wait a bit between captures
        time.sleep(config.DELAY_BETWEEN_CAPTURES)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished capturing images for {user_name}. Saved in: {user_dir}")
    return True

if __name__ == "__main__":
    name = input("Enter user name (or ID) to register: ").strip()
    if name:
        capture_user_images(name)
    else:
        print("Error: User name cannot be empty.")
