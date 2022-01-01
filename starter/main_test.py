import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def home():
    api = TestClient(app)
    return api


def test_get(home):
    connect = home.get("/")
    assert connect.status_code == 200
    assert connect.json() == ["Welcome"]


def test_post_1(home):
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
    })
    assert connect.status_code == 307


def test_post_2(home):
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
    assert connect.status_code == 422
