from utils.data_management.training_model_tree import *
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

    #run_eda(df, "Survived")

    #print(df.head())

    print("Training the model...")
    output = training_tree(df, "Survived")
    print("Best estimator:", output[0], "\n")
    print("accuracy on training set:", output[1], "\n")
    print("Accuracy:", output[2], "\n")
    print("Confusion Matrix:\n", output[3], "\n")
    print("F1 Score:", output[4], "\n")
    print("Precision:", output[5], "\n")
    print("Recall:", output[6], "\n")
    print("Classification Report:\n", output[7])
