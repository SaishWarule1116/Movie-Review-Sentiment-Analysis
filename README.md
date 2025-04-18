# Movie Review Sentiment Analysis
This project is a **Streamlit web application** that predicts the sentiment (Positive or Negative) of movie reviews using a Logistic Regression model trained on the IMDB dataset. The app preprocesses text using NLTK and TF-IDF vectorization, and it’s deployed on **Hugging Face Spaces** for public access.
# Features



Input a movie review and get a sentiment prediction with confidence score.



Example reviews to test the model.



Built with Python, Streamlit, Scikit-learn, and NLTK.


# Project Structure

movie-sentiment-app/

 **├── app.py**              # Main Streamlit app script
  
  **├── requirements.txt**    # Python dependencies
  
  **├── model.joblib**        # Pre-trained Logistic Regression model

  **├── vectorizer.joblib**   # Pre-trained TF-IDF vectorizer

  **├── README.md**           # Project documentation


# Model Details


  **Algorithm :** Logistic Regression


  **Dataset :** IMDB movie reviews (loaded via the datasets library during training)


  **Preprocessing :**

Text cleaning (lowercase, remove punctuation)

Tokenization and stopword removal (NLTK)

Lemmatization (NLTK)

TF-IDF vectorization (Scikit-learn)

  **Accuracy :** ~89% on the test set

# Demo

## Demo

Try out the sentiment analysis model live on Hugging Face: [Live Demo](https://huggingface.co/spaces/SaishWarule1116/Movie_Review_Sentiment_Analysis)
