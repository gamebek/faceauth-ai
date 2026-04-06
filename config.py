import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data Collection Settings
NUM_IMAGES_TO_CAPTURE = 10
DELAY_BETWEEN_CAPTURES = 0.5  # seconds

# Preprocessing Settings
IMAGE_SIZE = (64, 64)

# Model Settings
MODEL_PATH = os.path.join(MODELS_DIR, 'face_model.pkl')
DISTANCE_THRESHOLD = 40.0  # Threshold for KNN distance-based access control

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
