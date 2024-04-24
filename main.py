import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

st.title("Image Classification with Streamlit")

# Function to preprocess the image
def preprocess_image(image):
    image_size = 224

    # Resize the image to match the input shape of the model
    resized_image = image.resize((image_size, image_size))

    # Convert the image to array
    image_array = np.array(resized_image)

    # Normalize pixel values to range [0, 1]
    image_array = image_array / 255.0

    return image_array

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Load your model
    model_path = 'reg6.hdf5'
    weights_location = 'reg6_weights.hdf5'
    regnety006_custom_model = load_model(model_path)
    regnety006_custom_model.load_weights(weights_location)

    # Make prediction
    prediction = regnety006_custom_model.predict(np.expand_dims(processed_image, axis=0))

    return prediction

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image as PIL Image
    pil_image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(pil_image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(pil_image)

    # Display the predicted probabilities for each class
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # Modify according to your class names
    st.write("Predicted Probabilities:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {prediction[0][i]*100:.2f}%")

    # Assuming prediction is an array of probabilities for each class,
    # you can find the predicted label using argmax
    predicted_label = np.argmax(prediction)
    st.write("Predicted Label:", predicted_label)
