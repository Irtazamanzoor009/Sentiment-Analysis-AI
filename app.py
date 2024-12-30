import streamlit as st
import joblib
from scipy.sparse import hstack

# Load the pre-trained model and vectorizers
model = joblib.load("linear_svc_model.pkl")
vectorize_word = joblib.load("vectorize_word.pkl")
vectorize_char = joblib.load("vectorize_char.pkl")

# Title for the app
st.title("Sentiment Prediction App")
st.write("Predict review sentiment (Happy, Ok, or Unhappy) using the trained LinearSVC model.")

# Input box for review text
user_input = st.text_area("Enter a review text:", "")

# Button for prediction
if st.button("Predict"):
    if user_input:
        # Transform input using the saved vectorizers
        word_features = vectorize_word.transform([user_input])
        char_features = vectorize_char.transform([user_input])
        input_features = hstack([char_features, word_features])

        # Make prediction
        prediction = model.predict(input_features)

        # Display the result
        st.write("### Predicted Sentiment: ", prediction[0])
    else:
        st.warning("Please enter review text for prediction.")
