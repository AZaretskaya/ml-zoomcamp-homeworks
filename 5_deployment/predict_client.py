import requests

url = 'http://localhost:9696/predict'

client_id = 'from_question_4'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

response = requests.post(url, json=client).json()
print(response)

if response['give_card']:
    print(f'The client {client_id} will get a credit card.')
else:
    print(f'The client {client_id} will not get a credit card.')