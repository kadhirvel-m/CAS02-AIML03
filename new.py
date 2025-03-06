    import requests
    import json

    API_KEY = "AIzaSyAIGqO--WwGGA1jDvcq3U-CpOvzMAe2mZo"
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    BRAND_CONTEXT = {
        "kfc": "fried chicken, taste, delivery speed, pricing, offers",
        "nike": "shoes, sportswear, comfort, quality, pricing",
        "zomato": "food delivery, pricing, customer service, app experience",
        "swiggy": "delivery speed, app performance, customer service, pricing",
        "iphone": "camera, battery life, price, performance",
        "samsung": "smartphones, camera, battery, price, features"
    }

    def analyze_tweet_sentiment(brand):
        """
        Simulates sentiment data for different brands.
        """
        return {
            "positive": 70,
            "negative": 15,
            "neutral": 15,
            "top_concerns": ["High price", "Late delivery"],
            "top_praises": ["Quality product", "Fast service"]
        }

    def chat_with_stack_ai(user_input):
        """
        Analyzes tweet sentiment and generates brand-specific insights.
        """
        if "analyze" in user_input.lower():
            brand = user_input.split("analyze")[-1].strip().lower()
            sentiment = analyze_tweet_sentiment(brand)
            
            context = BRAND_CONTEXT.get(brand, "brand reputation, customer experience, pricing")

            insight_request = (
                f"{brand.capitalize()} Sentiment Data: Positive {sentiment['positive']}%, Negative {sentiment['negative']}%. "
                f"Key Complaints: {', '.join(sentiment['top_concerns'])}. "
                f"Customer Praise: {', '.join(sentiment['top_praises'])}. "
                f"Generate a short insight based on {context}."
            )

            data = {"contents": [{"parts": [{"text": insight_request}]}]}
            headers = {"Content-Type": "application/json"}
            response = requests.post(URL, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                try:
                    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return reply
                except KeyError:
                    return "Error: Unexpected format."
            else:
                return f"Error: {response.status_code}"

        return "Try: 'Analyze [Brand Name]' to get insights."

    if __name__ == "__main__":
        print("STACK AI - Twitter Sentiment Analysis (Type 'exit' to quit)")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            response = chat_with_stack_ai(user_input)
            print(f"STACK AI: {response}")
