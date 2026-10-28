
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Age Prediction")
st.write("Upload a picture or take a photo to estimate your age.")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age-model.keras")

model = load_model()

# uploading and taking a photo tabs
tab1, tab2 = st.tabs(["Upload Picture", "Take Photo"])

with tab1:
    uploaded_file = st.file_uploader("Choose a picture", type=["jpg", "jpeg", "png"], key="upload")

with tab2:
    camera_picture = st.camera_input("Take a picture", key="camera")

picture = uploaded_file if uploaded_file is not None else camera_picture

# Predicting
if picture is not None:
    image = Image.open(picture).convert("RGB")
    st.image(image, caption="Your picture")

    # Prepare the image for the model
    image = image.resize((48, 48))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image, verbose=0)
    age = float(prediction[0][0])
    st.success(f"Estimated age: {age:.0f} years")
