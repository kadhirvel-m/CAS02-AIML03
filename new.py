import requests
import json

API_KEY = "AIzaSyAIGqO--WwGGA1jDvcq3U-CpOvzMAe2mZo"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def analyze_sentiment(feedback):
    """Analyzes customer sentiment and returns classification, confidence score, and actionable insights."""
    
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Analyze the sentiment of the given customer feedback and classify it into:
    - Positive
    - Neutral
    - Negative

    Also provide:
    1. A confidence score (0 to 1).
    2. A brief explanation for classification.
    3. Suggested business actions based on the sentiment.

    Feedback: "{feedback}"
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        result = response.json()

        # Extracting response data
        response_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
        return response_text

    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

# Example Usage
if __name__ == "__main__":
    feedback = input("Enter customer feedback: ")
    sentiment_result = analyze_sentiment(feedback)
    print("\n🔍 Sentiment Analysis Result:\n", sentiment_result)