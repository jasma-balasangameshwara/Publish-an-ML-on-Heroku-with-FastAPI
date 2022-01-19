# Put the code for your API here.
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from starter.train_model import score, process_data, inference
from joblib import dump, load


app = FastAPI()


class Person(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    education: str = Field(..., example="Assoc-acdm")
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    os.system("dvc remote add -d s3-bucket s3://ml-heroku-fastapi-bucket")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")


@app.get("/")
def index():
    return {"Welcome Success"}


@app.post("/prediction")
async def predict_salary(data: Person):
    model = load(
        "starter/model/model.joblib")
    encoder = load(
        "starter/model/encoder.joblib")
    lb = load(
        "starter/model/lb.joblib")

    array = np.array([[data.age,
                       data.workclass,
                       data.education,
                       data.marital_status,
                       data.occupation,
                       data.relationship,
                       data.race,
                       data.sex,
                       data.hours_per_week,
                       data.native_country]])

    dataframe = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])
    x, _, _, _ = process_data(dataframe, training=False, encoder=encoder, lb=lb)

    prediction = inference(model, x)

    y = lb.inverse_transform(prediction)[0]
    return {"salary": y}
