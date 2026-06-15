import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("model/brain_tumor_unet.keras", compile=False)

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

st.title("🧠 Brain Tumor Segmentation using U-Net")
st.write("Upload an MRI scan and the model will detect tumor regions automatically.")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.subheader("Original MRI Image")
    st.image(image, use_column_width=True)

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image_resized = cv2.resize(image, (128, 128))

    # Normalize
    image_norm = image_resized / 255.0

    # Convert to 3-channel (because model expects (128,128,3))
    input_img = np.stack([image_norm]*3, axis=-1)

    # Add batch dimension
    input_img = np.expand_dims(input_img, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    pred = model.predict(input_img)[0]
    pred_mask = (pred.squeeze() > 0.5).astype(np.uint8)

    # -----------------------------
    # Tumor Area Calculation
    # -----------------------------
    tumor_pixels = np.sum(pred_mask)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    tumor_percentage = (tumor_pixels / total_pixels) * 100

    # -----------------------------
    # Visualization
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input (Resized)")
        st.image(image_resized, clamp=True)

    with col2:
        st.subheader("Predicted Mask")
        st.image(pred_mask * 255, clamp=True)

    with col3:
        st.subheader("Overlay")

        color_img = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        overlay = color_img.copy()

        overlay[pred_mask == 1] = [255, 0, 0]

        st.image(overlay)

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📊 Tumor Analysis Report")

    st.write(f"**Tumor Pixels:** {tumor_pixels}")
    st.write(f"**Tumor Percentage:** {tumor_percentage:.2f}%")

    if tumor_percentage > 10:
        st.warning("⚠ Large tumor region detected")
    elif tumor_percentage > 3:
        st.info("🟡 Moderate tumor region detected")
    else:
        st.success("🟢 Small / minimal tumor region detected")