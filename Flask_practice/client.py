import requests

url = "http://localhost:5000/analyze"
data = {
    "name": "Heather",
    "age": 30,
    "message": "Hello Flask!"
}

try:
    response = requests.post(url, json=data, timeout=5)  # 타임아웃 추가
    if response.status_code == 200:
        print("Response from server:", response.json())
    else:
        print("Failed:", response.status_code, response.text)
except requests.exceptions.ConnectionError:
    print("Error: Unable to connect to the server. Is the server running?")
except requests.exceptions.Timeout:
    print("Error: The request timed out.")
except Exception as e:
    print("An error occurred:", str(e))