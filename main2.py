import cv2
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import base64

from util import set_background

set_background('./bgs/bg5.png')

def preprocess_image(img):
    # Resize the image to (224, 224)
    resized_image = cv2.resize(img, (224, 224))
    
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    
    # Convert the LAB image back to BGR color space
    final_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    
    # Extract the green channel
    green_channel = final_image[:, :, 1]
    
    # Apply Gaussian blur to the green channel
    blurred_image = cv2.GaussianBlur(green_channel, (3, 3), 0)
    
    # Stack the blurred green channel to form the final processed image
    processed_image = np.stack([blurred_image] * 3, axis=-1)
    
    return processed_image

def classify(image, model, class_names):
    # Preprocess the image
    processed_image = preprocess_image(np.array(image))
    
    # Normalize the image
    normalized_image_array = (processed_image.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    
    # Determine the predicted class and confidence score
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# set title
st.title('DR classification')

# set header
st.header('Please upload a Retina image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/reg6.h5')

# load class names
class_names = ['NoDR', 'Mild', 'Moderate', 'Severe', 'ProliferativeDR']

# Define class meanings
class_meanings = {
    'NoDR': "Healthy Retina",
    'Mild': "Mild Non-proliferative Diabetic Retinopathy (NPDR) - Microaneurysms present",
    'Moderate': "Moderate NPDR - More severe signs, including microaneurysms, dot/blot hemorrhages, and cotton wool spots",
    'Severe': "Severe NPDR - Extensive hemorrhages and/or cotton wool spots in all 4 quadrants of the retina",
    'ProliferativeDR': "Proliferative Diabetic Retinopathy (PDR) - Neovascularization present, which can lead to vision loss"
}

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

    # Additional logic for displaying specific information based on predicted class
    if class_name in class_meanings:
        st.sidebar.warning("Detected Disease : " + class_name)
        st.sidebar.info(class_meanings[class_name])
    else:
        st.sidebar.warning("Unknown Disease Detected")
