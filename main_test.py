import pytest
from fastapi.testclient import TestClient
from main import app

home = TestClient(app)


def test_index():
    connect = home.get("/")
    assert connect.status_code == 200
    assert connect.json() == ["Welcome"]


def test_predict_salary_1():
    data = {
        "age": 32,
        "workclass": "Private",
        "education": "Assoc-acdm",
        "maritalStatus": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 50,
        "nativeCountry": "United-States"
    }
    connect = home.post("/prediction/", json=data)
    assert connect.status_code == 200


def test_predict_salary_2():
    connect = home.post("/prediction/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Assoc-acdm",
        "maritalStatus": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 50,
        "nativeCountry": "United-States"
    }).json()
    assert connect
