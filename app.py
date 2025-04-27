import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('pneumonia_model.h5')

# Sidebar content
st.sidebar.title("App Menu")
st.sidebar.subheader("About")
st.sidebar.markdown("""
    This is a **Pneumonia Detection** app that uses X-ray images to identify signs of pneumonia.
    The app utilizes a deep learning model trained on X-ray images to classify them as either **Normal** or **Pneumonia**.
""")

st.sidebar.subheader("Source")
st.sidebar.markdown("""
    The model was trained using publicly available **chest X-ray datasets**.
    - [Dataset Source](https://github.com/Omkar2703/Pneumonia-X-ray-detection.git)
""")

# Title with custom styling
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Pneumonia X-ray Detection</h1>", unsafe_allow_html=True)

# Description
st.markdown("<p style='text-align: center; font-size: 18px; color: #808080;'>Upload a chest X-ray image and get the result: Normal or Pneumonia.</p>", unsafe_allow_html=True)

# File uploader with custom styling
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# Displaying image and prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale

    # Make prediction
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "Pneumonia Detected ❌"
        result_color = "red"
    else:
        result = "Normal ✅"
        result_color = "green"
    
    # Display prediction with custom styling
    st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{result}</h2>", unsafe_allow_html=True)

    # Show more information or explanations (optional)
    if result == "Pneumonia Detected ❌":
        st.markdown("<p style='text-align: center; color: red;'>This X-ray suggests the presence of pneumonia. Please consult a medical professional for further diagnosis.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: green;'>No signs of pneumonia detected. Your lungs appear normal!</p>", unsafe_allow_html=True)

# Optional: Add custom CSS styling to the page
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 20px;
        font-weight: bold;
    }
    .stImage {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)
