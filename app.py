import streamlit as st
import sklearn
import helper
import pickle
import nltk
from PIL import Image

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("model.pkl",'rb'))

vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]{
    background-size: auto; /* Prevents scaling with window size */
    transform: scale(1); /* Zoom level set to 180% */
    transform-origin: center; /* Keeps the zoom centered */
    height: 100vh;
    width: 100vw;
    margin: 0;
    padding: 0;
    overflow: hidden; }

    [data-testid="stHeader"]{

    }
    .center-content {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh; /* Full viewport height */
            flex-direction: column;
            position: relative;
            z-index: 1;
        }}
        .stTextInput, .stButton {{
            margin-top: 20px;
            width: 300px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


title = "<h1 style='text-align: left; color: #FF5733; white-space: nowrap;'>Sentiment Analysis Application using ML ✨</h1>" 
st.markdown(title, unsafe_allow_html=True)

st.markdown('<div class="center-content">', unsafe_allow_html=True)

text = st.text_input("Please enter your review")

state = st.button("Predict","review")

token= helper.preprocess_text(text)
vector = vectorizer.transform([token])
prediction = model.predict(vector).item()

if state:
    # Add your prediction logic here
    if prediction == 1:

        st.markdown("<h3 style='background-color: white; color: green;padding: 15px 20px;border-radius: 20px;box-shadow: 0 4px 6px rgba(0, 0, 0, 1); /* Adds a subtle shadow for better appearance */text-align: right;font-family: Arial, sans-serif; font-size: 70px;display: inline-block'>👍 Positive Review</h3>",unsafe_allow_html=True)
        st.markdown(
        """
        <style>
        .dynamic-bg {
            animation: green-to-transparent 10s infinite;
            height: 100vh;
            width: 100vw;
            position: fixed;
            top: 0;
            left: 0;
            z-index: -1;
        }

        @keyframes green-to-transparent {
            0% { background-color: rgba(0, 77, 0, 0.8); } /* Dark Green with 80% opacity */
            25% { background-color: rgba(0, 128, 0, 0.6); } /* Green with 60% opacity */
            50% { background-color: rgba(102, 255, 102, 0.4); } /* Light Green with 40% opacity */
            75% { background-color: rgba(204, 255, 204, 0.2); } /* Pale Green with 20% opacity */
            100% { background-color: rgba(255, 255, 255, 0); } /* Fully Transparent */
        }
        </style>
        <div class="dynamic-bg"></div>
        """,
        unsafe_allow_html=True
    )
        st.balloons()
       
    else:
        st.markdown("<h3 style='background-color: white; color: red; padding: 15px 20px;border-radius: 20px;box-shadow: 0 4px 6px rgba(0, 0, 0, 1); /* Adds a subtle shadow for better appearance */text-align: right;font-family: Arial, sans-serif;font-size: 70px;display: inline-block''>👎 Negative Review</h3>",unsafe_allow_html=True)
        st.markdown(
        """
        <style>
        .dynamic-bg {
            animation: red-to-transparent 10s infinite;
            height: 100vh;
            width: 100vw;
            position: fixed;
            top: 0;
            left: 0;
            z-index: -1;
        }

        @keyframes red-to-transparent {
            0% { background-color: rgba(128, 0, 0, 0.8); } /* Dark Red with 80% opacity */
            25% { background-color: rgba(255, 0, 0, 0.6); } /* Red with 60% opacity */
            50% { background-color: rgba(255, 102, 102, 0.4); } /* Light Red with 40% opacity */
            75% { background-color: rgba(255, 204, 204, 0.2); } /* Pale Red with 20% opacity */
            100% { background-color: rgba(255, 255, 255, 0); } /* Fully Transparent */
        }
        </style>
        <div class="dynamic-bg"></div>
        """,
        unsafe_allow_html=True,
    )
