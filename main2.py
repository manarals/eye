import streamlit as st

# Define the theme
def my_theme():
    primaryColor = "#24475B"
    backgroundColor = "#3A3A3A"
    secondaryBackgroundColor = "#F3F3F3"
    textColor = "#FFFFFF"
    font = "sans-serif"

    # Apply the theme
    custom_theme = f"""
    [theme]
    primaryColor = "{primaryColor}"
    backgroundColor = "{backgroundColor}"
    secondaryBackgroundColor = "{secondaryBackgroundColor}"
    textColor = "{textColor}"
    font = "{font}"
    """

    st.write(custom_theme, unsafe_allow_html=True)

# Apply the theme
my_theme()

# Your Streamlit app code goes here...


# Now you can continue with the rest of your code
from keras.models import load_model
from PIL import Image
import numpy as np
import base64

from util import set_background, classify

#set_background('./bgs/bg5.png')

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
