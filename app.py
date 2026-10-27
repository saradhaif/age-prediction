
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Age Prediction 📸")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age_model.keras")

model = load_model()

# Open camera
picture = st.camera_input("Take a picture of your face")

if picture is not None:
    image = Image.open(picture).convert("RGB")
    st.image(image)

    # Prepare image for the model
    image = image.resize((48, 48))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict age
    prediction = model.predict(image, verbose=0)
    age = prediction[0][0]

    st.success(f"Your estimated age is {age:.0f} years!")
