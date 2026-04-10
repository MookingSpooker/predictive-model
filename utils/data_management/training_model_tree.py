import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from category_encoders.target_encoder import TargetEncoder 
import pandas as pd

def training_tree(df, target_column):
    # separating features and target variable
    X = df.drop(columns=[target_column])
    X = pd.get_dummies(X,prefix=["Sex", "Embarked"]) # one-hot encoding for categorical variables
    y = df[target_column]

    # splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7) # 80% train, 20% temp
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=7) # 10% val, 10% test
    #creating a pipeline for data preprocessing and model training

    #define the hyperparameters for the decision tree classifier

    parameters = {
        "criterion": ['gini', 'entropy',],
        "max_depth": [1,3,5,7,8,9,11,13,15,17,19],
        "min_samples_split": [2, 5, 10, 15, 20, 25, 30, 50],
        "min_samples_leaf": [1, 2, 4, 5,6,7,8,9,10,11, 16],
    }
    
    #initializing the decision tree classifier
    clf = DecisionTreeClassifier(random_state=7)

    RandomizedSearchCV_clf = RandomizedSearchCV(estimator=clf, param_distributions=parameters, n_iter=100, cv=5, random_state=7 ,scoring='f1')


    # training the model
    RandomizedSearchCV_clf.fit(X_train, y_train)

    # making predictions
    y_pred = RandomizedSearchCV_clf.predict(X_test)

    # calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    #f1_training = f1_score(y_train, RandomizedSearchCV_clf.predict(X_train), average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)


    return RandomizedSearchCV_clf.best_estimator_ ,RandomizedSearchCV_clf.best_score_ , accuracy, cm, f1, precision, recall , report #best score is the best f1 score on the training set