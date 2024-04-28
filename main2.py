import streamlit as st

# Define CSS styles for the theme
def set_custom_style():
    st.markdown(
        f"""
        <style>
            body {{
                color: #FFFFFF;
                background-color: #3A3A3A;
            }}
            .st-bk {{
                background-color: #F3F3F3;
            }}
            .st-br {{
                border-radius: 10px;
            }}
            .st-ch {{
                border-radius: 10px;
            }}
            .st-cp {{
                border-radius: 10px;
            }}
            .st-ct {{
                border-radius: 10px;
            }}
            .st-cv {{
                border-radius: 10px;
            }}
            .st-dh {{
                border-radius: 10px;
            }}
            .st-ej {{
                border-radius: 10px;
            }}
            .st-ex {{
                border-radius: 10px;
            }}
            .st-fx {{
                border-radius: 10px;
            }}
            .st-fy {{
                border-radius: 10px;
            }}
            .st-gg {{
                border-radius: 10px;
            }}
            .st-hc {{
                border-radius: 10px;
            }}
            .st-jo {{
                border-radius: 10px;
            }}
            .st-k {{
                border-radius: 10px;
            }}
            .st-kf {{
                border-radius: 10px;
            }}
            .st-l {{
                border-radius: 10px;
            }}
            .st-m {{
                border-radius: 10px;
            }}
            .st-n {{
                border-radius: 10px;
            }}
            .st-o {{
                border-radius: 10px;
            }}
            .st-q {{
                border-radius: 10px;
            }}
            .st-s {{
                border-radius: 10px;
            }}
            .st-t {{
                border-radius: 10px;
            }}
            .st-u {{
                border-radius: 10px;
            }}
            .st-v {{
                border-radius: 10px;
            }}
            .st-w {{
                border-radius: 10px;
            }}
            .st-y {{
                border-radius: 10px;
            }}
            .st-z {{
                border-radius: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_style()

# Now you can continue with the rest of your code
from keras.models import load_model
from PIL import Image
import numpy as np
import base64

from util import set_background, classify

set_background('./bgs/bg5.png')

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
