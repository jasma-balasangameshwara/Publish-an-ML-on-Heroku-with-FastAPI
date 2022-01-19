import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from joblib import dump, load
from starter.eda import clean
import scipy
import numpy as np
import logging


# Add the necessary imports for the starter code.

# Add code to load in the data.
def data_split():
    data = pd.read_csv("cleaned_census.csv")
    train_set, test = train_test_split(data, test_size=0.20)
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    return data, train_set, test, categorical_features


def process_data(train, training=True, encoder=None, lb=None, label=None):
    _, _, _, categorical_features = data_split()
    if label is not None:
        y_train = train["salary"]
        train = train.drop(["salary"], axis=1)
    else:
        y_train = np.array([])

    x_categorical = train[categorical_features].values
    x_continuous = train.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y_train = lb.fit_transform(y_train.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y_train = lb.transform(y_train.values).ravel()
        except AttributeError:
            pass
    x_train = np.concatenate([x_continuous, x_categorical], axis=1)

    return x_train, y_train, encoder, lb


def model_train():
    _, train, test, categorical_features = data_split()
    x_train, y_train, encoder, lb = process_data(train, training=True, label="salary")
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    dump(model,
         "model/model.joblib")
    dump(encoder,
         "model/encoder.joblib")
    dump(lb,
         "model/lb.joblib")


def score(test, model, encoder, lb):
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    sliced = []
    for each_category in categorical_features:
        for index in test[each_category].unique():
            unique_df = test[test[each_category] == index]
            x_test, y_test, _, _ = process_data(unique_df, training=False, encoder=encoder, lb=lb, label="salary")

            pred_y = model.predict(x_test)

            fbeta = fbeta_score(y_test, pred_y, beta=1, zero_division=1)
            precision = precision_score(y_test, pred_y, zero_division=1)
            recall = recall_score(y_test, pred_y, zero_division=1)

            print(fbeta, precision, recall)
            logging.info(fbeta, precision, recall)
            line = str(each_category) + str(index) + str(fbeta) + str(precision) + str(recall)
            sliced.append(line)
    with open('data/raw/slice_output.txt', 'w') as out:
        for value in sliced:
            out.write(value + '\n')


def inference(model, data):
    preds_y = model.predict(data)
    return preds_y


if __name__ == '__main__':
    model_train()
    _, _, tests, _ = data_split()
    model1 = load(
        "model/model.joblib")
    encoder1 = load(
        "model/encoder.joblib")
    lb1 = load(
        "model/lb.joblib")
    score(tests, model1, encoder1, lb1)

