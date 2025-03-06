import re
import tweepy
import spacy
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Twitter API Credentials
api_key = "7MED7RVYolgx2S035hPTO2pQ8"
api_secret_key = "ytnv7lYwxeGUfv4qLFnJALRIiflcyoZI8Iqfx9iDk30Du5oNbK"
access_token = "937181091449864192-cdaxAnSe2oCPn63RCifIRNCfg0uSQvb"
access_token_secret = "0zKbGVHwXVlDx65OA8mFNxVjEufzhQXEOM0lvlLXmV5kC"

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis", model="roberta-base")


def fetch_tweets(query, count=10):
    """
    Fetches tweets based on a query, limiting the request to a maximum of 10 tweets.
    """
    count = min(count, 10)  # Ensure the count does not exceed 10
    try:
        tweets = api.search_tweets(q=query, count=count, tweet_mode="extended")
        tweet_texts = [tweet.full_text for tweet in tweets]
        print(f"Fetched Tweets for '{query}': {tweet_texts}")  # Debugging line
        return tweet_texts
    except tweepy.errors.TweepyException as e:  # Fixed error handling
        print(f"Error fetching tweets: {e}")
        return []


def clean_tweet(tweet):
    """
    Cleans a tweet by removing URLs, mentions, hashtags, and special characters.
    """
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    return tweet.strip()


def preprocess_tweet(tweet):
    """
    Processes a tweet by tokenizing and lemmatizing while removing stop words.
    """
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)


def analyze_sentiment(tweet):
    """
    Analyzes the sentiment of a tweet using the sentiment analysis pipeline.
    """
    result = sentiment_pipeline(tweet)
    return result[0]["label"]


def aggregate_sentiments(tweets):
    """
    Aggregates sentiment counts from a list of tweets.
    """
    sentiments = [analyze_sentiment(tweet) for tweet in tweets]
    sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    return sentiment_counts


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_counts = None
    tweet_data = []

    if request.method == "POST":
        user_input = request.form.get("keyword")
        print(f"User Input: {user_input}")  # Debugging line

        if user_input:
            tweets = fetch_tweets(user_input)
            if tweets:
                cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
                print(f"Cleaned Tweets: {cleaned_tweets}")  # Debugging line

                preprocessed_tweets = [preprocess_tweet(tweet) for tweet in cleaned_tweets]
                print(f"Preprocessed Tweets: {preprocessed_tweets}")  # Debugging line

                sentiment_counts = aggregate_sentiments(preprocessed_tweets)
                print(f"Sentiment Counts: \n{sentiment_counts}")  # Debugging line

                for tweet in preprocessed_tweets:
                    sentiment = analyze_sentiment(tweet)
                    tweet_data.append({"Tweet": tweet, "Sentiment": sentiment})

    return render_template("index.html", sentiment_counts=sentiment_counts, tweet_data=tweet_data)

if __name__ == "__main__":
    app.run(debug=True)
