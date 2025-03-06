import tweepy

# Twitter API v2 credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAABYFzwEAAAAAtzx0rUr7NSWnd5pLnGdApJuQBIA%3Dm7CTNGGM182ASFN6WDGEBDrdw9pnG449Z9NXEZxo1xnAIlCAHf"

client = tweepy.Client(bearer_token=bearer_token)

def fetch_tweets(query, count=10):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=["text"])
        if tweets.data:
            return [tweet.text for tweet in tweets.data]
        else:
            return []
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        return []
