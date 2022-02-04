# API code
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field
from starter.train_model import inference_dict
from fastapi.encoders import jsonable_encoder
from joblib import load

app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country", ]


class Person(BaseModel):
    age: int = Field(..., example=56)
    workclass: str = Field(..., example="Local-gov")
    education: str = Field(..., example="Bachelors")
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Tech-support")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")


@app.get("/")
def index():
    return {"Welcome Success"}


@app.post("/")
async def predict_salary(data: Person):
    data_json = jsonable_encoder(data)
    model = load(
        "starter/model/model.joblib")
    encoder = load(
        "starter/model/encoder.joblib")
    lb = load(
        "starter/model/lb.joblib")

    '''array = np.array([[data.age,
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
    x, _, _, _ = process_data(dataframe, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    prediction = inference(model, x)
    '''
    prediction = inference_dict(data_json, model, encoder, lb, cat_features)
    if prediction == '0':
        pred = '<=50K'
    else:
        pred = '>50K'

    return {"income": pred}
