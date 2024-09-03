import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


# Load the model and vectorizer
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)


    with open("vectorizer.pkl", "rb") as f:
        loaded_vectorizer = pickle.load(f)
    # Download NLTK data
    nltk.download("punkt")
    nltk.download("stopwords")
    return loaded_model,loaded_vectorizer




# Function to remove stopwords and tokenize the input text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stopwords_list = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords_list]
    # Join the tokens back into a string
    processed_text = " ".join(filtered_tokens)
    print('processed text',processed_text)
    return processed_text

# Function to transform user input data
def transform_data(review_text,loaded_vectorizer):
    # Preprocess the review text
    processed_review = preprocess_text(review_text)
    # Combine name and preprocessed review
    transformed_data =  loaded_vectorizer.transform([processed_review])
    return transformed_data

# Streamlit App
def main():
    loaded_model,loaded_vectorizer = load_models()
    st.title("Sentiment Analysis")

    # Input fields
    user_review = st.text_area("Your Review")

    if st.button("Predict Sentiment"):
        transformed_data = transform_data(user_review,loaded_vectorizer)
        prediction = loaded_model.predict(transformed_data)
        if(prediction[0] == 1):
            st.write("Predicted sentiment: Positive")
        elif(prediction[0] == 2):
            st.write("Predicted sentiment: Negative")
        else:
            st.write("Predicted sentiment: Neutral")

        

if __name__ == "__main__":
    main()
