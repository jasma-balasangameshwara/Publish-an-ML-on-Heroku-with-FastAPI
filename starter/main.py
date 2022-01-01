# Put the code for your API here.
import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
from joblib import load

from starter.train_model import score, model_train

app = FastAPI()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Person(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    education: str = Field(..., example="Assoc-acdm")
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")


@app.get("/")
def index():
    return {"Welcome"}


@app.post('/prediction')
async def predict_income(inputrow: Person):
    row_dict = jsonable_encoder(inputrow)
    prediction = score(row_dict)

    return {"salary": prediction}
