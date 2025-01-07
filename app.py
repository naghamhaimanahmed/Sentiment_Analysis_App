import streamlit as st
import pickle
import nltk
import helper   # Ensure the helper module is in your project

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

# Custom styles
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        color: #555;
    }
    .prediction-box {
        font-size: 24px;
        font-weight: bold;
        color: #FFA726;
        text-align: center;
        border: 2px solid #FFA726;
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
        background-color: #FFF3E0;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 40px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.markdown('<div class="main-title">Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter a review to predict its sentiment using Machine Learning</div>', unsafe_allow_html=True)

# Input text
text = st.text_area("Please enter your review:", placeholder="Type your review here...", height=150)

# Predict sentiment
if st.button("Predict Sentiment"):
    if text.strip():
        # Preprocess and predict
        token=helper.preprocess_text(text)
        vectorized_data = vectorizer.transform([token])
        prediction = model.predict(vectorized_data)

        # Display the prediction result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.markdown(f'<div class="prediction-box">Predicted Sentiment: {sentiment}</div>', unsafe_allow_html=True)
    else:
        st.error("Please enter some text before predicting.")

# Footer
st.markdown('<div class="footer">Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)
