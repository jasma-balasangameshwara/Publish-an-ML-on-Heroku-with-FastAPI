import requests

data = {
    "age": 56,
    "workclass": "Local-gov",
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post(
    url='https://ml-heroku-fastapi.herokuapp.com/',
    json=data
)

print(response.status_code)
print(response.json())
