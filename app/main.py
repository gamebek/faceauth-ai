"""
Entry point for Streamlit application.
Run this using: streamlit run app/main.py
"""
import os
import sys
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from app.utils import model_exists, registered_users

# Page configuration
st.set_page_config(
    page_title="FaceAuth AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5em;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1em 0;
    }
    .status-card {
        background-color: #ffffff;
        padding: 1em;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5em 0;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9em;
        color: #666;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

# Quick stats in sidebar
users = registered_users()
st.sidebar.subheader("📊 Quick Stats")
st.sidebar.metric("Registered Users", len(users))
st.sidebar.metric("Model Status", "Trained" if model_exists() else "Not Trained")

st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.rerun()

st.sidebar.markdown("### Authentication")
if st.sidebar.button("📝 Register", use_container_width=True):
    st.switch_page("pages/register.py")

if st.sidebar.button("🔐 Login", use_container_width=True):
    st.switch_page("pages/login.py")

st.sidebar.markdown("---")
st.sidebar.markdown("### Development")
if st.sidebar.button("📊 Model Evaluation", use_container_width=True):
    from src.evaluate import evaluate_model
    st.sidebar.text("Running evaluation...")
    try:
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            evaluate_model()
        output = f.getvalue()
        st.sidebar.code(output, language="text")
        st.sidebar.success("Evaluation complete!")
    except Exception as e:
        st.sidebar.error(f"Evaluation failed: {str(e)}")

# Main content
st.markdown('<h1 class="main-header">🛡️ FaceAuth AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Secure face-based authentication system built with machine learning</p>', unsafe_allow_html=True)

# Feature overview
st.header("🚀 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📸 Face Registration</h4>
        <p>Capture and store face images for user registration with live camera preview.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 ML-Powered Recognition</h4>
        <p>K-Nearest Neighbors algorithm for accurate face matching with confidence scoring.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🔒 Secure Authentication</h4>
        <p>Distance-based threshold prevents false positives and ensures security.</p>
    </div>
    """, unsafe_allow_html=True)

# Current status
st.header("📊 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.markdown(f"""
    <div class="status-card">
        <div class="metric-value">{len(users)}</div>
        <div class="metric-label">Registered Users</div>
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    model_status = "✅ Ready" if model_exists() else "⚠️ Not Trained"
    color = "#28a745" if model_exists() else "#ffc107"
    st.markdown(f"""
    <div class="status-card">
        <div class="metric-value" style="color: {color};">{"Trained" if model_exists() else "Pending"}</div>
        <div class="metric-label">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

with status_col3:
    st.markdown(f"""
    <div class="status-card">
        <div class="metric-value">{config.DISTANCE_THRESHOLD}</div>
        <div class="metric-label">Security Threshold</div>
    </div>
    """, unsafe_allow_html=True)

# Quick start guide
st.header("🎯 Quick Start")

tab1, tab2, tab3 = st.tabs(["📝 Registration", "🔐 Authentication", "⚙️ Configuration"])

with tab1:
    st.markdown("""
    1. **Navigate to Register page** using the sidebar
    2. **Enter your name/ID** in the text field
    3. **Enable camera preview** to adjust your position
    4. **Capture face images** (5-20 recommended)
    5. **Train the model** to create your face recognition profile
    """)

with tab2:
    st.markdown("""
    1. **Navigate to Login page** using the sidebar
    2. **Enable camera preview** to position yourself
    3. **Click Authenticate** when ready
    4. **Wait for face detection and analysis**
    5. **View authentication result** with confidence score
    """)

with tab3:
    st.markdown(f"""
    **Current Configuration:**
    - Image Size: {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} pixels
    - Images per user: {config.NUM_IMAGES_TO_CAPTURE}
    - Security threshold: {config.DISTANCE_THRESHOLD}
    - Model: K-Nearest Neighbors (K=3)

    **Directories:**
    - Raw images: `{config.RAW_DATA_DIR}`
    - Models: `{config.MODELS_DIR}`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1em;">
    <p>Built with ❤️ using OpenCV, scikit-learn, and Streamlit</p>
    <p><small>FaceAuth AI - Secure face-based authentication system</small></p>
</div>
""", unsafe_allow_html=True)
