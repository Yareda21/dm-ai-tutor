import requests

url = "http://localhost:8001/embed"
headers = {"Content-Type": "application/json"}
data = {"texts": ["tell me about the owner of the school?"]}

try:
    response = requests.post(url, headers=headers, json=data)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except ValueError:
    print("Failed to parse JSON response")
    print("Raw response:", response.text)