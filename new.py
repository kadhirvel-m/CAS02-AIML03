import requests
import json

API_KEY = "AIzaSyAIGqO--WwGGA1jDvcq3U-CpOvzMAe2mZo"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def chat_with_gemini(user_input):
    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        try:
            reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return reply
        except KeyError:
            return "Error: Unexpected response format."
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    print("Simple Gemini Chatbot (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = chat_with_gemini(user_input)
        print(f"Gemini: {response}")
