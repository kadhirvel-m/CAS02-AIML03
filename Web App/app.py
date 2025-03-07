from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests

app = Flask(__name__)

API_KEY = "AIzaSyA--oVDvnjuNy255wvfiRkknotM_U8P_IA"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def fetch_feedback(brand_name):
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Generate 5-10 customer feedback examples (both positive and negative) for the brand "{brand_name}".
    Ensure diversity in feedback content.
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        feedback_list = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No feedback generated.")
        return feedback_list.split("\n")
    except requests.exceptions.RequestException as e:
        return [f"API Error: {e}"]

def analyze_sentiment(brand_name):
    feedback_list = fetch_feedback(brand_name)
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Analyze the sentiment of the following customer feedback for {brand_name}:
    {feedback_list}

    Classify each as Positive, Neutral, or Negative.
    Provide:
    1. Overall sentiment (Positive/Negative/Neutral)
    2. A short overall insight
    3. Positive factors
    4. Negative factors
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        sentiment_analysis = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No analysis.")
        return sentiment_analysis
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    brand_name = request.json.get("brand_name")
    if not brand_name:
        return jsonify({"error": "Brand name is required"}), 400
    result = analyze_sentiment(brand_name)
    return jsonify({"analysis": result})

@app.route("/result")
def result():
    analysis = request.args.get('analysis')
    return render_template("result.html", analysis=analysis)

if __name__ == "__main__":
    app.run(debug=True)
