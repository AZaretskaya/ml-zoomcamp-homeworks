import requests


url = 'http://localhost:9696/predict'

user_id = 'user_1'
user = {"daily_time_spent_on_site": 59.21,
        "age": 35,
        "area_income": 73347.67,
        "daily_internet_usage": 144.62,
        "city": "Lake Beckyburgh",
        "gender": "Male",
        "country": "Liechtenstein"
         }


response = requests.post(url, json=user)
prediction = response.json()


if prediction['click_on_ad']:
    print(f"The user '{user_id}' clicked on an advertisement with probability {prediction['click_probability']}.")
else:
    print(f"The user '{user_id}' didn't clicked on an advertisement. Clicking probability is {prediction['click_probability']}.")