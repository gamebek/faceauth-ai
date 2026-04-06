import os
import sys
import time
import streamlit as st
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from src.predict import login_with_face
from app.utils import model_exists

st.set_page_config(page_title="Login | FaceAuth AI")

st.title("Login with Face")
st.write("Authenticate using your registered face.")

# Check prerequisites
if not model_exists():
    st.error("❌ No trained model found. Please register and train the model first.")
    st.stop()

# Camera preview section
st.subheader("Camera Preview")
col1, col2 = st.columns([2, 1])

with col1:
    show_preview = st.checkbox("Show camera preview", value=True)

with col2:
    if show_preview:
        preview_placeholder = st.empty()
        stop_button = st.button("Stop Preview")

# Live preview
if show_preview and not stop_button:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        st.info("📹 Position your face in the camera. Click 'Stop Preview' when ready to authenticate.")
        while show_preview and not stop_button:
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.1)
            else:
                st.error("Failed to capture from camera")
                break
        cap.release()
    else:
        st.error("Could not access webcam")

# Authentication section
st.subheader("Face Authentication")
st.write("Click the button below to authenticate with your face.")

if st.button("🔐 Authenticate", type="primary", use_container_width=True):
    with st.spinner("Scanning your face... Please look directly at the camera."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing camera...")
        progress_bar.progress(20)

        status_text.text("Detecting face...")
        progress_bar.progress(50)

        status_text.text("Analyzing features...")
        progress_bar.progress(80)

        # Perform authentication
        label, distance, message = login_with_face()

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # Display result
        if label:
            st.success(f"✅ {message}")
            st.balloons()
        else:
            st.error(f"❌ {message}")

        # Show technical details
        with st.expander("🔍 Technical Details"):
            if distance is not None:
                st.write(f"**Distance to nearest match:** {distance:.2f}")
                st.write(f"**Threshold:** {config.DISTANCE_THRESHOLD}")

                # Visual distance indicator
                if distance <= config.DISTANCE_THRESHOLD:
                    st.progress(min(1.0, 1.0 - (distance / config.DISTANCE_THRESHOLD)))
                    st.write("🟢 **Match confidence:** High")
                else:
                    st.progress(min(1.0, distance / (config.DISTANCE_THRESHOLD * 2)))
                    st.write("🔴 **Match confidence:** Low")
            else:
                st.write("No distance information available")

# Quick actions
st.markdown("---")
st.subheader("Quick Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Retrain Model", help="Retrain the model with current user data"):
        from src.train import train_model
        with st.spinner("Retraining model..."):
            try:
                train_model()
                st.success("✅ Model retrained successfully!")
            except Exception as e:
                st.error(f"❌ Retraining failed: {str(e)}")

with col2:
    if st.button("📊 View Model Stats", help="Show model performance metrics"):
        from src.evaluate import evaluate_model
        with st.spinner("Evaluating model..."):
            try:
                # Capture output
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    evaluate_model()
                output = f.getvalue()

                st.code(output, language="text")
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")

# Help section
with st.expander("ℹ️ Login Tips"):
    st.markdown("""
    **Authentication Tips:**
    - Ensure good lighting and face the camera directly
    - Keep a neutral expression
    - Stay still during scanning
    - Make sure your face is clearly visible

    **Troubleshooting:**
    - If authentication fails, try different lighting or angles
    - Ensure you are registered and the model is trained
    - Check that no other applications are using the webcam
    - Try refreshing the page if issues persist

    **Security Notes:**
    - The system uses distance-based matching with a configurable threshold
    - Lower distance values indicate better matches
    - The threshold prevents false positives
    """)
