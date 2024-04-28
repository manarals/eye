import streamlit as st
from keras.models import load_model
from PIL import Image

# Import util functions from local file
from util import set_background, classify

# Load the theme from config.toml
st.set_theme('./config.toml')

# Set title and header
st.title('DR classification')
st.header('Please upload a Retina image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/reg6.h5')

# Load class names
class_names = ['NoDR', 'Mild', 'Moderate', 'Severe', 'ProliferativeDR']

# Define class meanings
class_meanings = {
    'NoDR': "Healthy Retina",
    'Mild': "Mild Non-proliferative Diabetic Retinopathy (NPDR) - Microaneurysms present",
    'Moderate': "Moderate NPDR - More severe signs, including microaneurysms, dot/blot hemorrhages, and cotton wool spots",
    'Severe': "Severe NPDR - Extensive hemorrhages and/or cotton wool spots in all 4 quadrants of the retina",
    'ProliferativeDR': "Proliferative Diabetic Retinopathy (PDR) - Neovascularization present, which can lead to vision loss"
}

# Display image and classification result
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

    # Additional logic for displaying specific information based on predicted class
    if class_name in class_meanings:
        st.sidebar.warning("Detected Disease : " + class_name)
        st.sidebar.info(class_meanings[class_name])
    else:
        st.sidebar.warning("Unknown Disease Detected")
