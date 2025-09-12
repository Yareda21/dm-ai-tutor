import requests

url = 'http://127.0.0.1:8000/ai/chat'
while True:
    prompt = input("You: ")

    data = {'prompt': prompt, 'use_rag': True}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print("ai:", response.json().get('answer', 'No response field in JSON'))
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")