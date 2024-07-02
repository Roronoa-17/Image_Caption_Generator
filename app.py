import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from util import generate_caption

# Function to load the model

# Streamlit app
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    caption = generate_caption(image)
    st.write(f"Caption: {caption}")

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info("This app uses a Deep Learning model(RNN model) along with VGG16 model(feature extractor) to generate captions for uploaded images.")
st.sidebar.info("Upload an image to get started!")
st.sidebar.info("The model is trained on Flickr8k dataset.")
st.sidebar.info("By Priyesh Gawali")
st.sidebar.markdown("[Github repository](https://github.com/Roronoa-17/Image_Caption_Generator.git)")