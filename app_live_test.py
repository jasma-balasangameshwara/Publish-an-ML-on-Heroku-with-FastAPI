import requests

data = {
    "age": 47,
    "workclass": "Private",
    "education": "Assoc-acdm",
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Male",
    "hours_per_week": 55,
    "native_country": "United-States",
}

response = requests.post(
    url='https://ml-heroku-fastapi.herokuapp.com/prediction',
    json=data
)

