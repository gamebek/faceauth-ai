"""
Helper functions for the UI layer.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config


def registered_users():
    if not os.path.exists(config.RAW_DATA_DIR):
        return []

    return sorted([
        folder for folder in os.listdir(config.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATA_DIR, folder))
    ])


def model_exists():
    return os.path.exists(config.MODEL_PATH)


def get_user_image_count(user_name):
    """Get the number of images captured for a specific user."""
    user_dir = os.path.join(config.RAW_DATA_DIR, user_name)
    if not os.path.exists(user_dir):
        return 0

    return len([
        f for f in os.listdir(user_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def get_total_images():
    """Get the total number of images across all users."""
    users = registered_users()
    return sum(get_user_image_count(user) for user in users)


def delete_user(user_name):
    """Delete a user's data (use with caution)."""
    import shutil
    user_dir = os.path.join(config.RAW_DATA_DIR, user_name)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        return True
    return False


def get_system_stats():
    """Get comprehensive system statistics."""
    users = registered_users()
    total_images = get_total_images()
    model_trained = model_exists()

    stats = {
        "users_count": len(users),
        "total_images": total_images,
        "model_trained": model_trained,
        "images_per_user": {user: get_user_image_count(user) for user in users},
        "config": {
            "image_size": config.IMAGE_SIZE,
            "num_images_to_capture": config.NUM_IMAGES_TO_CAPTURE,
            "distance_threshold": config.DISTANCE_THRESHOLD,
        }
    }

    return stats
