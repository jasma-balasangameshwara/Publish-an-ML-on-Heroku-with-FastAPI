import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def home():
    homeq = TestClient(app)
    return homeq


def test_index(home):
    connect = home.get("/")
    assert connect.status_code == 200
    assert connect.json() == ["Welcome Success"]


def test_predict_salary_1(home):
    connect = home.post("/prediction", json={
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States"
    })
    assert connect.status_code == 200
    assert connect.json() == {"salary": "<=50K"}


def test_predict_salary_2(home):
    connect = home.post("/prediction", json={
        "age": 30,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "India"
    })
    assert connect.status_code == 200
    assert connect.json() == {"salary": ">50K"}
