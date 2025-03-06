import re
import tweepy
import spacy
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

api_key = "lrR71NHrb40mfwJ1wkhzhlmpg"
api_secret_key = "3idSPOsmi6hZHARiYRWJax8LpKfr5z3mQDgl5zkV9Y6iKvEHJq"
access_token = "1897499259722981376-CbppY1UAGYrSxlqdUkGTSymMSpRSOj"
access_token_secret = "AehDdjMtXBXoAWvv2Iuvm6RWpTAFylSqcwvhJYLDiUr7G"

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis", model="roberta-base")


def fetch_tweets(query, count=100):
    try:
        tweets = api.search_tweets(q=query, count=count, tweet_mode="extended")
        return [tweet.full_text for tweet in tweets]
    except tweepy.TweepError as e:
        return []


def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    return tweet.strip()


def preprocess_tweet(tweet):
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)


def analyze_sentiment(tweet):
    result = sentiment_pipeline(tweet)
    return result[0]["label"]


def aggregate_sentiments(tweets):
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

        if user_input:
            tweets = fetch_tweets(user_input)
            if tweets:
                cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
                preprocessed_tweets = [preprocess_tweet(tweet) for tweet in cleaned_tweets]

                sentiment_counts = aggregate_sentiments(preprocessed_tweets)

                for tweet in preprocessed_tweets:
                    sentiment = analyze_sentiment(tweet)
                    tweet_data.append({"Tweet": tweet, "Sentiment": sentiment})

    return render_template("index.html", sentiment_counts=sentiment_counts, tweet_data=tweet_data)


if __name__ == "__main__":
    app.run(debug=True)
