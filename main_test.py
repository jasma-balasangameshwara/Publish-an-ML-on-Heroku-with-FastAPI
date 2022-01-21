import pytest
from fastapi.testclient import TestClient
from main import app



home = TestClient(app)


def test_index():
    connect = home.get("/")
    assert connect.status_code == 200
    assert connect.json() == ["Welcome Success"]


def test_predict_salary_1():
    data1 = {
        "age": 19,
        "workclass": "Private",
        "education": "HS-grad",
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    connect = home.post('/', json=data1)
    assert connect.status_code == 200
    assert connect.json() == {"income": '<=50K'}


def test_predict_salary_2():
    data2 = {
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
    connect = home.post('/', json=data2)
    assert connect.status_code == 200
    assert connect.json() == {"income": '>50K'}
