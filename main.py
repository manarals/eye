import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (image_size, image_size))
    normalized_image = image / 255.0
    return normalized_image

# Function to predict the class of the image
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_label = np.argmax(prediction)
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # Modify according to your class names
    predicted_class = class_names[predicted_label]
    confidence = prediction[0][predicted_label]
    return predicted_class, confidence

# Load the model
image_size = 224
weights_location = './model/reg6_weights.h5'
model_path = './model/reg6.h5'

try:
    model = load_model(model_path)
    model.load_weights(weights_location)
except Exception as e:
    st.error("Error loading model: " + str(e))

# Streamlit app
st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predicted_class, confidence = predict_image(image)
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)
