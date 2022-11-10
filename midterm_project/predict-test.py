import requests


url = 'http://localhost:9696/predict'

house_id = 'test_house'
house = {"relative_compactness": 0.71,
        "surface_area": 710.50,
        "wall_area": 269.50,
        "roof_area": 220.50,
        "overall_height": 3.50,
        "orientation": 2.00,
        "glazing_area": 0.40,
        "glazing_area_distribution": 1.00
         }


response = requests.post(url, json=house).json()
prediction = round(response['heating_load'], 2)

print(f"The predicted heating load for the house '{house_id}' is {prediction} kWh/m.")