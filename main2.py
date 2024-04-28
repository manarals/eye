import streamlit as st

# Define the theme
def my_theme():
    primaryColor = "#24475B"
    backgroundColor = "#3A3A3A"
    secondaryBackgroundColor = "#F3F3F3"
    textColor = "#FFFFFF"
    font = "sans-serif"

    # Apply the theme
    st.markdown(
        f"""
        <style>
            /* Streamlit App Main Style */
            body {{
                color: {textColor};
                background-color: {backgroundColor};
                font-family: {font};
            }}
            .stApp {{
                background-color: {backgroundColor};
            }}

            /* Streamlit Widgets Style */
            .stTextInput > div > div > input {{
                color: {textColor};
                background-color: {secondaryBackgroundColor};
                border-color: {primaryColor};
            }}
            .stTextInput > div > label {{
                color: {textColor};
            }}

            .stButton > button {{
                color: {textColor};
                background-color: {primaryColor};
            }}
            .stButton > button:hover {{
                background-color: {secondaryBackgroundColor};
            }}
            .stButton > button:active {{
                background-color: {secondaryBackgroundColor};
                color: {primaryColor};
            }}

            /* Streamlit Markdown Style */
            .stMarkdown {{
                color: {textColor};
            }}
            .stMarkdown a {{
                color: {primaryColor};
            }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the theme
my_theme()

# Your Streamlit app code goes here...
