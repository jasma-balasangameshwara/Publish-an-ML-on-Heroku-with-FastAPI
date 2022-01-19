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
    r = home.post("/prediction", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}


def test_predict_salary_2(home):
    connect = home.post("/prediction", json={
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
    })
    assert connect.status_code == 200
    assert connect.json() == {"salary": "<=50K"}
