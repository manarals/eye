import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

st.title("Image Classification with Streamlit")

# Function to preprocess the image
def preprocess_image(image):
    image_size = 224

    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (image_size, image_size))

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

    # Convert LAB image back to BGR
    final_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Extract the green channel
    green_channel = final_image[:, :, 1]

    # Apply Gaussian Blur to the green channel
    blurred_image = cv2.GaussianBlur(green_channel, (3, 3), 0)

    # Stack the blurred green channel to form a 3-channel image
    processed_image = np.stack([blurred_image] * 3, axis=-1)

    return processed_image

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Load your model
    model_path = 'reg6.h5'
    weights_location = 'reg6_weights.h5'
    regnety006_custom_model = load_model(model_path)
    regnety006_custom_model.load_weights(weights_location)

    # Make prediction
    prediction = regnety006_custom_model.predict(np.expand_dims(processed_image, axis=0))

    return prediction

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(image)

    # Display the predicted probabilities for each class
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # Modify according to your class names
    st.write("Predicted Probabilities:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {prediction[0][i]*100:.2f}%")

    # Assuming prediction is an array of probabilities for each class,
    # you can find the predicted label using argmax
    predicted_label = np.argmax(prediction)
    st.write("Predicted Label:", predicted_label)
