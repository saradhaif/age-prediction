
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("Age Prediction 📸")
st.write("Upload a picture or take a photo to estimate your age.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age_model.keras")

model = load_model()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

tab1, tab2 = st.tabs(["Upload Picture", "Take Photo"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a picture",
        type=["jpg", "jpeg", "png"],
        key="upload"
    )

with tab2:
    camera_picture = st.camera_input("Take a picture", key="camera")

picture = uploaded_file if uploaded_file is not None else camera_picture

if picture is not None:
    image = Image.open(picture).convert("RGB")
    image_array = np.array(image)

    st.image(image_array, caption="Your picture")

    # Detect the face
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No face detected. Try a clearer picture with your face facing the camera.")

    else:
        # Select the largest face
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

        face = image_array[y:y+h, x:x+w]

        # Resize the face for the model
        face = Image.fromarray(face).resize((48, 48))
        face = np.array(face).astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict age
        prediction = model.predict(face, verbose=0)
        age = float(prediction[0][0])

        st.success(f"Estimated age: {age:.0f} years")
        st.caption("This is an AI estimate and may not reflect actual age.")
