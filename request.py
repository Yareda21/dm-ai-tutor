import requests

res = requests.post(
    "http://127.0.0.1:8000/ai/chat",
    json={"prompt": "What is digital marketing?", "use_rag": True}
)
print(res.json())
