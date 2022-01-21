import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from joblib import dump, load
from starter.eda import clean
import scipy
import numpy as np
import logging


# Add the necessary imports for the starter code.

# Add code to load in the data.
def data_split():
    data = pd.read_csv("cleaned_census.csv", index_col=None)
    train_set, test = train_test_split(
        data, test_size=0.20, random_state=8, shuffle=True)
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


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.
    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        # print('-----', X_categorical.shape, X_continuous.shape, X.shape)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            y = None

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder, lb


def model_train():
    _, train, test, _ = data_split()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    x_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, training=True,
                                                 label="salary")
    model = RandomForestClassifier(
        random_state=8, max_depth=16, n_estimators=128)
    model.fit(x_train, y_train)
    dump(model,
         "model/model.joblib")
    dump(encoder,
         "model/encoder.joblib")
    dump(lb,
         "model/lb.joblib")


def score(test, model, encoder, lb):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    sliced = []
    for each_category in cat_features:
        for index in test[each_category].unique():
            unique_df = test[test[each_category] == index]
            x_test, y_test, _, _ = process_data(unique_df, categorical_features=cat_features, training=False,
                                                encoder=encoder, lb=lb, label="salary")

            pred_y = model.predict(x_test)

            fbeta = fbeta_score(y_test, pred_y, beta=1, zero_division=1)
            precision = precision_score(y_test, pred_y, zero_division=1)
            recall = recall_score(y_test, pred_y, zero_division=1)

            # print(fbeta, precision, recall)
            logging.info(fbeta, precision, recall)
            line = str(each_category) + str(index) + str(fbeta) + str(precision) + str(recall)
            sliced.append(line)
    with open('data/raw/slice_output.txt', 'w') as out:
        for value in sliced:
            out.write(value + '\n')


def inference_dict(data, cat_features):
    model = load(
        "starter/model/model.joblib")
    encoder = load(
        "starter/model/encoder.joblib")
    lb = load(
        "starter/model/lb.joblib")


    X_categorical = list()
    X_continuous = list()

    for key, value in data.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_category = encoder.transform([X_categorical])
    y_continuous = np.asarray([X_continuous])

    row_transformed = list()
    row_transformed = np.concatenate([y_continuous, y_category], axis=1)
    preds_in = model.predict(row_transformed)
    y = lb.inverse_transform(preds_in)[0]
    return str(y)


def inference(model, data):
    preds_y = model.predict(data)
    return '>50K' if preds_y[0] else '<=50K'


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
