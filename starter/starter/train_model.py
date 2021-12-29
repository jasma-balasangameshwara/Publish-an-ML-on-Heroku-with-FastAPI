# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
from starter.eda import clean
import scipy
import numpy as np


# Add the necessary imports for the starter code.

# Add code to load in the data.
def data_split():
    data = clean(
        "D:/CAREER TRANSITION/Udacity/MLOps - Nanodegree/Capstone Projects/FastAPI/nd0821-c3-starter-code-master/starter/data/raw/census.csv",
        "D:/CAREER TRANSITION/Udacity/MLOps - Nanodegree/Capstone Projects/FastAPI/nd0821-c3-starter-code-master/starter/data/processed/cleaned_census.csv")
    train_set, test = train_test_split(data, test_size=0.20)
    return data, train_set, test


def train_predict():
    data, train, test = data_split()
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    y_train = train["salary"]
    train = train.drop(["salary"], axis=1)
    x_categorical = train[categorical_features].values
    x_continuous = train.drop(*[categorical_features], axis=1)
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    lb = LabelBinarizer()
    x_categorical = encoder.fit_transform(x_categorical)
    y_train = lb.fit_transform(y_train.values).ravel()
    x_train = np.concatenate([x_continuous, x_categorical], axis=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    y_predicted = model.predict(test)
    dump(model,
         "D:/CAREER TRANSITION/Udacity/MLOps - Nanodegree/Capstone Projects/FastAPI/nd0821-c3-starter-code-master/starter/model/model.joblib")
    return model, y_predicted
