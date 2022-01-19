import requests

data = {
    "age": 30,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "India"
}

response = requests.post(
    url='https://ml-heroku-fastapi.herokuapp.com/',
    json=data
)

print(response.status_code)
print(response.json())
print(response.content)
