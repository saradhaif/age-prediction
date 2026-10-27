
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Age Prediction 📸")
st.write("Upload a picture to estimate your age.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age-model.keras")

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a picture",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded picture")

    image = image.resize((48, 48))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
    age = prediction[0][0]

    st.success(f"Estimated age: {age:.0f} years")
