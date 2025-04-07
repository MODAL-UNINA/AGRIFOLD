#%%
import requests
import json

# API Key
api_key = 'your API key'

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

def chat_with_gemini(user_message, conversation_history=[]):

    conversation_history.append({"text": user_message})

    payload = {
        "contents": [
            {
                "parts": conversation_history
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }


    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:

        response_data = response.json()
        model_reply = response_data['candidates'][0]['content']['parts'][0]['text']

        print("\nGemini's response:")
        print(model_reply)

        conversation_history.append({"text": model_reply})

        return conversation_history
    else:
        print(f"Error {response.status_code}: {response.text}")
        return conversation_history

def start_chat():
    conversation_history = []

    print("Welcome to the Gemini chatbot!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            break

        conversation_history = chat_with_gemini(user_message, conversation_history)

start_chat()



# %% Example questions:

# 1. My tomato leaves affect Powdery Mildews, how to treat it?

# 2. Your answer is very helpful, thank you.

# 3. What should I do if my squashis infected with powdery mildew? Does it need the same treatment?

# 4. What should I do if my tomato leaves are late blight?

# 5. Can you recommend some websites?