import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os

from util import classify, set_background

# Get absolute path to the directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set absolute path for model and background image
model_path = os.path.join(current_dir, 'model', 'reg6.h5')
background_image_path = os.path.join(current_dir, 'bgs', 'bg5.png')

# Set background
set_background(background_image_path)

# Set title
st.title('Diabetic Retinopathy Detection')

# Set header
st.header('Please upload a Retina image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model(model_path)

# Load class names
with open(os.path.join(current_dir, 'model', 'labels.txt'), 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
