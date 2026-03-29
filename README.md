# FaceAuth AI

A real-world AI-powered authentication system that allows users to register and log in using their face. 

This project simulates modern authentication systems and focuses on building a complete ML product, from data collection to deployment via Streamlit UI.

## Project Structure

```bash
faceauth-ai/
├── app/                        # Streamlit application (UI layer)
│   ├── main.py                 # Entry point (run this file)
│   ├── pages/                  # Streamlit pages
│   │   ├── register.py
│   │   └── login.py
│   └── utils.py                # Helper functions for UI
├── data/                       # Dataset
│   ├── raw/                    # Original captured images
│   └── processed/              # Preprocessed images
├── models/                     # Saved ML models
│   └── face_model.pkl
├── src/                        # Core ML logic
│   ├── data_collection.py      # Webcam capture logic
│   ├── preprocessing.py        # Face detection, resizing, cleaning
│   ├── feature_engineering.py  # Flattening, transformations
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation metrics
│   └── predict.py              # Inference logic
├── notebooks/                  # Optional experimentation (Jupyter)
│   └── exploration.ipynb
├── tests/                      # Unit tests
│   └── test_pipeline.py
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Ignore unnecessary files
└── config.py                   # Configurations (paths, parameters)
```

## Setup Instructions

1. **Clone the repository** (if using Git):
   ```bash
   git clone <repo_url>
   cd faceauth-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Face Registration (Data Collection)**:
   Run the data collection script to register users and save their face images.
   ```bash
   python src/data_collection.py
   ```
   *Follow the prompt to enter your name and look at the webcam. It captures 10 images.*

4. **Face Preprocessing**:
   To test preprocessing on saved images:
   ```bash
   python src/preprocessing.py
   ```

## Next Steps for Project Development

- Finish `src/feature_engineering.py` (flatten images, create X, y).
- Implement `src/train.py` (train a KNN model, save using pickle).
- Build the `app/` using Streamlit for user-friendly Registration and Login.
