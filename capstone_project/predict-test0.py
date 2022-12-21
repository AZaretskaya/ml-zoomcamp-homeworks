import requests


url = 'http://localhost:9696/predict'

user_id = 'user_0'
user = {"daily_time_spent_on_site": 83.48,
        "age": 31,
        "area_income": 59340.99,
        "daily_internet_usage": 222.72,
        "city": "Williamport",
        "gender": "Male",
        "country": "Philippines"
        }


response = requests.post(url, json=user)
prediction = response.json()


if prediction['click_on_ad']:
    print(f"The user '{user_id}' clicked on an advertisement with probability {prediction['click_probability']}.")
else:
    print(f"The user '{user_id}' didn't clicked on an advertisement. Clicking probability is {prediction['click_probability']}.")