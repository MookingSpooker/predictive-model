from utils.data_management.data_analysis import *
from utils.data_management.metric_calculator import *
from utils.data_management.training_model import *

import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

if __name__ == '__main__':
    # populating age with median
    populate_nan_columns(df, "Age", find_column_median(df, "Age"))

    # populating embarked location with mode
    populate_nan_columns(df, "Embarked", find_column_mode(df, "Embarked"))

    # removing redundant columns
    remove_columns(df, ["PassengerId", "Name", "Ticket", "Cabin"])

    remove_duplicates(df)

    run_eda(df, "Survived")

    print(df.head())

    print("Training the model...")
    output = train_model(df, "Survived")
    print("Best score (training set): " + str(output[0]))
    print("Best Score (test set): " + str(output[1]))
    print("Predictions: " + str(output[2]))
    print("Prediction Probabilities: " + str(output[3]))