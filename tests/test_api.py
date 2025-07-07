import requests

payload = {
    "strategy": "2",
    "ticker": "TSLA",
    "testPeriod": "2",
    "customStart": ""
}

try:
    response = requests.post("http://localhost:5000/api/backtest", json=payload)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error communicating with backend:", e) 