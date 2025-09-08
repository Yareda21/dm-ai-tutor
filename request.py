import requests
url = "https://dm-ai-tutor.onrender.com/"
r = requests.post(url, json={"prompt":"Hello","session_id":"student-123"})
print(r.json())   # -> {"answer": "...", "raw": ...}