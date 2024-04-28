import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


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
# Load class names
class_names = []
with open('./model/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) >= 2:  # Check if line has at least two parts
            class_names.append(parts[1])
        else:
            # Handle improperly formatted line (optional)
            print(f"Ignoring improperly formatted line: {line}")

# No need to explicitly close the file, as it's automatically closed when exiting the 'with' block


# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
