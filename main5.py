import streamlit as st
st.set_page_config(
    page_title="MoqlatAI DR classification",
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
    predicted_probabilities = "\n".join([f"\n{key}: {value}\n" for key, value in predicted_probabilities.items()])

# Display predicted probabilities in the sidebar

    # Additional logic for displaying specific information based on predicted class
    if class_name in class_meanings:
        st.sidebar.title("Diabetic retinopathy")
        st.sidebar.write("A progressive eye disease that can lead to vision loss. It occurs when diabetes damages the blood vessels of the retina, the light-sensitive tissue at the back of the eye.")
        st.sidebar.image("./DR2.png", use_column_width=True)
        st.sidebar.success("Detected Class : " + class_name)
        st.sidebar.error("Prediction Probability: {}%".format(int(conf_score * 1000) / 10))
        st.sidebar.info(class_meanings[class_name])
        st.sidebar.warning("Predicted Probabilities:\n" + predicted_probabilities)

    else:
        st.sidebar.warning("Unknown Disease Detected")
def handle_user_input(user_input):
    if user_input.lower() in ["yes", "hi", "please", "chat", "hello", "salam", "bot"]:
        st.write("Hello there!\nPlease choose a number so I can assist you...")
        st.write("1. What is MoqlatAI")
        st.write("2. What is Diabetic Retinopathy")
        st.write("3. What are the stages of Diabetic Retinopathy")
    elif user_input == "1":
        st.write("**MoqlatAI** is an AI-powered website that utilizes advanced deep learning algorithms to detect Diabetic Retinopathy.\n\n **Mission:** \n"
          "Our mission is to contribute to Saudi Arabia's Vision 2030 by integrating technology into the healthcare sector, driving societal transformational success,"
          "that allows you to flex and grow your business with the best solutions.\n\n **Vision:** \n Our vision is to empower healthcare professionals with advanced disease detection tools, ensuring exceptional patient care"
          "and well-being.")
    elif user_input == "2":
        st.write("**Diabetic retinopathy** is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina)."
          " In some cases, diabetic retinopathy can progress to a severe stage where it causes vision loss or even blindness.\n Regular eye exams and managing blood sugar levels are essential for preventing and managing diabetic retinopathy.")
        
    elif user_input == "3":
        st.write("1- **Normal Retina:** In a healthy retina, the blood vessels are typically well-formed and function properly to supply oxygen and nutrients to the retinal tissue. There are no signs of swelling, leakage, or abnormal growth of blood vessels. The macula, the central part of the retina responsible for central vision, is flat and thin, allowing for clear vision.\n"
          "\n\n2- **Mild Nonproliferative Retinopathy:** The first stage of diabetic retinopathy involves the development of small areas of swelling in the retinal blood vessels, known as microaneurysms. These microaneurysms may cause minor leakage of fluid into the retina, leading to mild retinal swelling.\n"
          "\n\n3- **Moderate Nonproliferative Retinopathy:** As the disease progresses, more blood vessels may become blocked, resulting in areas of the retina being deprived of oxygen (ischemia). This stage is characterized by a greater extent of retinal damage compared to mild nonproliferative retinopathy.\n"
          "\n\n4- **Severe Nonproliferative Retinopathy:** In this stage, a significant number of retinal blood vessels are blocked, leading to widespread ischemia in the retina. The lack of oxygen triggers the growth of new, abnormal blood vessels (neovascularization), which marks the transition to the proliferative stage.\n"
          "\n\n5- **Proliferative Retinopathy:** This advanced stage is characterized by the growth of abnormal blood vessels into the vitreous, which is the gel-like substance filling the center of the eye. These fragile blood vessels are prone to bleeding, causing vitreous hemorrhage and potentially leading to sudden vision loss.")
    else:
        st.write("Stay safe!")

        
st.title("") 
st.title("")
st.title("")# Adds a horizontal rule for separation
st.subheader(":gray[Chat Assistant]  :speech_balloon:  :robot_face:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Hi, human! Is there anything I can help you with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle user input and display assistant response in chat message container
    with st.chat_message("assistant"):
        handle_user_input(prompt)
