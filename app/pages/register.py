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
from src.data_collection import capture_user_images
from src.train import train_model
from app.utils import model_exists, registered_users

st.set_page_config(page_title="Register | FaceAuth AI")

st.title("Register New User")
st.write("Create a new user profile by capturing face images and training the authentication model.")

# User input
name = st.text_input("Enter Name or ID", placeholder="e.g., john_doe")

# Camera preview section
st.subheader("Camera Preview")
col1, col2 = st.columns([2, 1])

with col1:
    run_preview = st.checkbox("Show camera preview", value=False)

with col2:
    if run_preview:
        preview_placeholder = st.empty()
        stop_preview = st.button("Stop Preview")

if run_preview and not stop_preview:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        st.info("Adjust your position and ensure good lighting. Click 'Stop Preview' when ready.")
        while run_preview and not stop_preview:
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            else:
                st.error("Failed to capture from camera")
                break
        cap.release()
    else:
        st.error("Could not access webcam")

# Image capture settings
st.subheader("Image Capture Settings")
num_images = st.slider(
    "Number of images to capture",
    min_value=5,
    max_value=20,
    value=config.NUM_IMAGES_TO_CAPTURE,
    step=1,
    help="More images generally improve recognition accuracy"
)

# Capture and train buttons
col1, col2 = st.columns(2)

with col1:
    capture_button = st.button("📸 Capture Face Images", type="primary", use_container_width=True)

with col2:
    train_button = st.button("🧠 Train Model", use_container_width=True)

# Capture images
if capture_button:
    if not name.strip():
        st.error("❌ Please enter a name or ID before capturing images.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Opening webcam and capturing images..."):
            status_text.text("Initializing camera...")
            progress_bar.progress(10)

            result = capture_user_images(name.strip(), num_images=num_images)

            if result:
                progress_bar.progress(100)
                status_text.text("✅ Images captured successfully!")
                st.success(f"📁 Captured {num_images} images for '{name.strip()}'")
                st.info("💡 Now click 'Train Model' to create the face recognition model.")
                time.sleep(2)  # Brief pause to show success
            else:
                progress_bar.progress(0)
                status_text.text("❌ Failed to capture images")
                st.error("❌ Failed to capture images. Please check your webcam and try again.")

        progress_bar.empty()
        status_text.empty()

# Train model
if train_button:
    with st.spinner("Training face recognition model..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Loading training data...")
        progress_bar.progress(20)

        try:
            train_model()
            progress_bar.progress(100)
            status_text.text("✅ Model trained successfully!")
            st.success("🎉 Model training complete! You can now use the Login page.")
            time.sleep(2)
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Training failed")
            st.error(f"❌ Model training failed: {str(e)}")

        progress_bar.empty()
        status_text.empty()

# Status section
st.markdown("---")
st.subheader("📊 Current Status")

col1, col2 = st.columns(2)

with col1:
    users = registered_users()
    if users:
        st.success(f"👥 Registered Users: {len(users)}")
        with st.expander("View Users"):
            for user in users:
                st.write(f"• {user}")
    else:
        st.info("👤 No users registered yet")

with col2:
    if model_exists():
        st.success("🤖 Model Status: Trained & Ready")
    else:
        st.warning("⚠️ Model Status: Not trained")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **Registration Tips:**
    - Ensure good lighting and face the camera directly
    - Keep a neutral expression
    - Move your head slightly between captures for better training
    - Use a unique name/ID for each user

    **Troubleshooting:**
    - If camera doesn't work, check browser permissions
    - Make sure no other applications are using the webcam
    - Try refreshing the page if issues persist
    """)
