import sklearn as sk
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from category_encoders.target_encoder import TargetEncoder 
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import accuracy_score ,confusion_matrix, f1_score, precision_score, recall_score, classification_report

def train_model(df, target_column):
    # separating features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7) # 80% train, 20% temp
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=7) # 10% val, 10% test
    #creating a pipeline for data preprocessing and model training
    estimators = [
        ('encoder', TargetEncoder()),                   #standerd prepsocessing step for categorical variables it will  convert them into numerical values based on the target variable
        ('clf', XGBClassifier(random_state=7))
    ]
    pipeline = sk.pipeline.Pipeline(steps=estimators)


    #set up hyperparameter tuning using Bayesian optimization
    
    search_space = {
        "clf__n_estimators": Integer(100, 1200),
        "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "clf__max_depth": Integer(2, 8),
        "clf__min_child_weight": Integer(1, 12),
        "clf__gamma": Real(0.0, 10.0),
        "clf__subsample": Real(0.6, 1.0),
        "clf__colsample_bytree": Real(0.6, 1.0),
        "clf__reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
        "clf__reg_lambda": Real(1e-6, 30.0, prior="log-uniform"),
        "clf__max_delta_step": Integer(0, 10),
        "clf__scale_pos_weight": Real(0.8, 3.0)
    }

    opt = BayesSearchCV(estimator=pipeline, search_spaces=search_space, n_iter=50, cv=3, scoring='f1', random_state=7)

    #train the XGBoost model:

    opt.fit(X_train, y_train)

    y_pred = opt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    #f1_training = f1_score(y_train, RandomizedSearchCV_clf.predict(X_train), average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    return opt.best_estimator_ ,opt.best_score_ , accuracy, cm, f1, precision, recall , report #best score is the best f1 score on the training set