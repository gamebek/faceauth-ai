# FaceAuth AI

A real-world AI-powered authentication system that allows users to register and log in using facial recognition instead of a password.

This project simulates a complete ML product with data collection, preprocessing, model training, evaluation, and a professional Streamlit UI with enhanced user experience features.

## Project Structure

```bash
faceauth-ai/
├── app/
│   ├── main.py              # Enhanced landing page with navigation and stats
│   ├── pages/
│   │   ├── register.py      # Face registration with camera preview and progress
│   │   └── login.py         # Authentication with live preview and technical details
│   └── utils.py             # Extended utility functions for system stats
├── data/
│   ├── raw/                 # Raw captured face images
│   └── processed/           # Processed feature data (generated)
├── models/
│   └── face_model.pkl       # Trained KNN model
├── src/
│   ├── data_collection.py   # Webcam image capture
│   ├── preprocessing.py     # Face detection and normalization
│   ├── feature_engineering.py # Data loading and preprocessing
│   ├── train.py             # Model training pipeline
│   ├── evaluate.py          # Model evaluation and metrics
│   └── predict.py           # Real-time face authentication
├── notebooks/
│   └── exploration.ipynb    # Data exploration notebook
├── tests/
│   └── test_pipeline.py     # Unit tests for core functionality
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
└── config.py               # Configuration settings
```

## 🚀 Key Features

### Face Registration
- **Live Camera Preview**: Real-time webcam feed with face detection overlay
- **Progress Tracking**: Visual progress bars during image capture
- **Automatic Training**: One-click model training after registration
- **User Management**: Support for multiple users with organized data storage

### Face Authentication
- **Live Authentication**: Real-time face recognition with confidence scoring
- **Security Threshold**: Configurable distance threshold to prevent false positives
- **Technical Details**: Display of detection confidence, processing time, and match distance
- **Quick Actions**: Direct access to model retraining and evaluation

### Professional UI
- **Modern Design**: Clean, responsive interface with custom styling
- **Navigation Sidebar**: Easy access to all features and system statistics
- **Status Dashboard**: Real-time metrics and system health indicators
- **Help Sections**: Built-in guidance and troubleshooting tips

### ML Pipeline
- **Robust Preprocessing**: Haar Cascade face detection with normalization
- **Feature Engineering**: Automated data loading and preprocessing pipeline
- **KNN Classification**: Distance-based face matching with configurable parameters
- **Model Evaluation**: Comprehensive metrics including accuracy and confusion matrix

## Setup Instructions

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd faceauth-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app/main.py
   ```

4. **Use the application:**
   - **Home Page**: View system status, features overview, and quick start guide
   - **Register Page**: Enter username, capture face images with live preview, train model
   - **Login Page**: Authenticate using face recognition with real-time feedback

## 🔧 Configuration

Key settings in `config.py`:
- `IMAGE_SIZE`: Face image dimensions (default: 64x64)
- `NUM_IMAGES_TO_CAPTURE`: Images per user during registration (default: 10)
- `DISTANCE_THRESHOLD`: Authentication security threshold (default: 40.0)
- `MODEL_PATH`: Location of trained model file

## 📊 Running Core Scripts

### Manual Data Collection
```bash
python src/data_collection.py
```

### Model Training
```bash
python src/train.py
```

### Model Evaluation
```bash
python src/evaluate.py
```

### Live Authentication (Command Line)
```bash
python src/predict.py
```

## 🧪 Testing

Run unit tests to validate core functionality:
```bash
python -m pytest tests/test_pipeline.py -v
# or
python tests/test_pipeline.py
```

## 📈 System Architecture

### Data Flow
1. **Registration**: Webcam capture → Face detection → Image storage
2. **Training**: Load images → Preprocessing → Feature extraction → KNN training
3. **Authentication**: Live capture → Face detection → Feature matching → Distance calculation

### Technologies Used
- **Computer Vision**: OpenCV with Haar Cascade classifiers
- **Machine Learning**: scikit-learn K-Nearest Neighbors
- **Web Framework**: Streamlit with custom CSS styling
- **Data Processing**: NumPy for image arrays
- **Testing**: pytest for unit testing

## 🔒 Security Features

- **Distance Threshold**: Prevents false positive authentications
- **Face Detection**: Only processes images containing detected faces
- **Model Persistence**: Secure storage of trained models
- **User Isolation**: Separate data storage per registered user

## 📝 Notes

- Raw face images are stored in `data/raw/<username>/`
- Processed features are cached in `data/processed/`
- Trained model is saved as `models/face_model.pkl`
- The system supports multiple concurrent users
- Camera permissions required for registration and authentication

## 🚀 Future Enhancements

- User management interface (delete users, view stored images)
- Model versioning and performance tracking
- Advanced face detection models (CNN-based)
- Multi-factor authentication integration
- Cloud deployment and API endpoints
- Mobile application support

---

**Built with ❤️ using OpenCV, scikit-learn, and Streamlit**
