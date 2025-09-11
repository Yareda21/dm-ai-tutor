import requests
resp = requests.post("http://127.0.0.1:8000/ai/chat",
                     json={"prompt":"What is digital marketing?","use_rag":True})
print(resp.status_code)
print(resp.json())
