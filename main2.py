import streamlit as st
st.set_page_config(
    page_title="DR classification",
    page_icon=":eye:",
    layout="centered",
    initial_sidebar_state="auto"
)


# Now you can continue with the rest of your code
from keras.models import load_model
from PIL import Image
import numpy as np
import base64

from util import set_background, classify

#set_background('./bgs/bg5.png')

# set title
st.markdown("<h1 style='color: #24475B;'>Diabatic Retinopathy classification</h1>", unsafe_allow_html=True)


# set header
st.markdown("<h3 style='color: #24475B;'>Please upload a Retina image</h2>", unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
#weights = load_weights('./model/reg6_weights.h5')
model = load_model('./model/reg6.h5')

# load class names
#class_names = ['NoDR', 'Mild', 'Moderate', 'Severe', 'ProliferativeDR']
class_names = []
with open('./model/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) >= 2:  # Check if line has at least two parts
            class_names.append(parts[1])
        else:
            # Handle improperly formatted line (optional)
            print(f"Ignoring improperly formatted line: {line}")

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
    class_name, conf_score, predicted_probabilities = classify(image, model, class_names)

        # write classification
    #st.markdown("<h2 style='color: #24475B;'>{}</h2>".format(class_name), unsafe_allow_html=True)
    #st.markdown("<h3 style='color: #24475B;'>score: {}%</h3>".format(int(conf_score * 1000) / 10), unsafe_allow_html=True)
    predicted_probabilities = "\n".join([f"{key}: {value}" for key, value in predicted_probabilities.items()])

# Display predicted probabilities in the sidebar

    # Additional logic for displaying specific information based on predicted class
    if class_name in class_meanings:
        st.sidebar.title("Diabetic retinopathy")
        st.sidebar.write("A progressive eye disease that can lead to vision loss. It occurs when diabetes damages the blood vessels of the retina, the light-sensitive tissue at the back of the eye.")
        st.sidebar.image("./DR2.png", use_column_width=True)
        st.sidebar.success("Detected class : " + class_name)
        st.sidebar.warning("Accuracy: {}%".format(int(conf_score * 1000) / 10))
        st.sidebar.info(class_meanings[class_name])
        st.sidebar.error("Predicted Probabilities:" + predicted_probabilities)

    else:
        st.sidebar.warning("Unknown Disease Detected")
