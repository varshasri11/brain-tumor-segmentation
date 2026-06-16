import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

st.title("🧠 Brain Tumor Segmentation")
st.write("Upload an MRI image to visualize preprocessing steps.")

uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    # Resize
    resized = cv2.resize(image_np, (128, 128))

    # Normalize
    normalized = resized / 255.0

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_np, caption="Original MRI Image", use_container_width=True)

    with col2:
        st.image(
            normalized,
            caption="Preprocessed MRI Image",
            use_container_width=True
        )

    st.success("Image preprocessing completed successfully!")

    st.info(
        "Note: This cloud demo shows image preprocessing only. "
        "The trained U-Net model requires TensorFlow, which is not currently supported "
        "by the Streamlit Cloud environment being used."
    )

else:
    st.write("Please upload an MRI image to begin.")