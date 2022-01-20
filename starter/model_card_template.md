# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
It is Gradient Boosting Classifier.

## Intended Use
This model predicts the salary of a person.

## Training Data
80% of the census.csv data is used for training.

## Evaluation Data
20% of the census.csv data is used for testing.

## Metrics
_Precision, Recall and fbeta are the metrics used to measure the model performance. For one of the iteration of the for loop, fbeta = 0.71, precision = 0.73, recall = 0.50 

## Ethical Considerations
Data related to race and gender. 

## Caveats and Recommendations
Some column names have '-' symbol in the data files. This has to be taken care of while referring to category feature names. Also, data had leading whitespaces. Data cleaning was performed to remove whitespaces.
