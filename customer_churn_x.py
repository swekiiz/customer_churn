from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaning and Feature Engineering
df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors="coerce")

df = df.dropna()

# drp_cols = ['Unnamed: 0', 'customerID']
drp_cols = ["customerID"]

df.drop(columns=drp_cols, inplace=True)

for col in ["InternetService", "PaymentMethod"]:
    dummies = pd.get_dummies(df[col], prefix=col).astype("int")
    df = df.join(dummies)
    df = df.drop(columns=[col])

gnd_val = {"Male": 0, "Female": 1}
yn_val = {"No": 0, "Yes": 1}
srv_val = {"No": 0, "Yes": 1, "No internet service": 2}
phn_val = {"No": 0, "Yes": 1, "No phone service": 2}
cnt_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# Payment Method and Internet Services are not Ordinal Categorical Variables so it speared the Label Encoding

df["gender"] = df["gender"].map(gnd_val).astype("int")
df["MultipleLines"] = df["MultipleLines"].map(phn_val).astype("int")
df["Contract"] = df["Contract"].map(cnt_val).astype("int")

# Yes/No
for i in df.columns:
    if df[i].dtype == "object" and df[i].nunique() == 2:
        if i != "gender":
            df[i] = df[i].map(yn_val).astype("int")

for i in df.columns:
    if df[i].dtype == "object" and df[i].nunique() == 3:
        if i != "Contract" and i != "InternetService" and i != "PaymentMethod":
            df[i] = df[i].map(srv_val).astype("int")

# Distribution
print(df.groupby("Churn").size())

# ## Save Clean Data
# df.to_csv("cleaned_telecom_users.csv", index = False)

##
# Machine Learning
##

# Evaluation Model


def evaluate_model(clf, X_test, y_test, model_name, oversample_type):
    print("--------------------------------------------")
    print("Model ", model_name)
    print("Data Type ", oversample_type)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    print(classification_report(y_test, y_pred))
    print("F1 Score ", f1)
    print("Recall ", recall)
    print("Precision ", precision)
    return [model_name, oversample_type, f1, recall, precision]


X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# Model Selection for Classification Set

models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
# append() takes exactly one argument (2 given)
# models.append("KNN", KNeighborsClassifier())
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

# Evaluate Model Peroformance

results = []
names = []
pretty_print = list()
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    pretty_print.append((cv_results.mean(), cv_results.std()))
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()), "\n")

##
# Handling Imbalanced TARGET Sets
##

# SMOTE

smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

# ADASYN

ada = ADASYN(random_state=130)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)

# SMOTE + Tomek Links
smtom = SMOTETomek(random_state=139)
X_train_smtom, y_train_smtom = smtom.fit_resample(X_train, y_train)

## SMOTE + ENN
smenn = SMOTEENN()
X_train_smenn, y_train_smenn = smenn.fit_resample(X_train, y_train)


def evaluate_model(clf, X_test, y_test, model_name, oversample_type):
    print("--------------------------------------------")
    print("Model ", model_name)
    print("Data Type ", oversample_type)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    print(classification_report(y_test, y_pred))
    print("F1 Score ", f1)
    print("Recall ", recall)
    print("Precision ", precision)
    return [model_name, oversample_type, f1, recall, precision]


models = {
    "DecisionTrees": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LinearSVC": LinearSVC(random_state=0),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "SGD": SGDClassifier(random_state=42),
    "CART": DecisionTreeClassifier(random_state=42),
    "LR": LogisticRegression(),
}

oversampled_data = {
    "ACTUAL": [X_train, y_train],
    "SMOTE": [X_train_sm, y_train_sm],
    "ADASYN": [X_train_ada, y_train_ada],
    "SMOTE_TOMEK": [X_train_smtom, y_train_smtom],
    "SMOTE_ENN": [X_train_smenn, y_train_smenn],
}

final_output = []
for model_k, model_clf in models.items():
    for data_type, data in oversampled_data.items():
        model_clf.fit(data[0], data[1])
        final_output.append(
            evaluate_model(model_clf, X_validation, y_validation, model_k, data_type)
        )

final_df = pd.DataFrame(
    final_output, columns=["Model", "DataType", "F1", "Recall", "Precision"]
).sort_values(by="F1", ascending=False)
print("Model Performance for Balanced and Imbalanced Datasets", end="")
print()
print(final_df)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 50, 100],
    "min_samples_split": [0, 5, 10],
    "min_samples_leaf": [0, 5, 10],
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05, 0.1],
    "max_features": ["auto", "sqrt", "log2"],
    "oob_score": ["True", "False"]
    # 'n_estimators': [0, 400, 800, 1000, 2000]
}

rfc = RandomForestClassifier(random_state=42)
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=2)
rfc_cv.fit(X_train, y_train)

params = rfc_cv.best_params_
print(params)

# Results
# {'criterion': 'entropy',
#  'max_depth': 50,
#  'max_features': 'auto',
#  'min_samples_leaf': 5,
#  'min_samples_split': 5,
#  'min_weight_fraction_leaf': 0.0,
#  'oob_score': 'True'}


pipe = make_pipeline(
    MinMaxScaler(),
    RandomForestClassifier(
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
        oob_score=params["oob_score"],
    ),
)

model = pipe.fit(X_train, y_train)
y_pred = model.predict(X_validation)
print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
print(
    f"Validation Accuracy Score: {model.score(X_validation, y_validation) * 100:.1f}%"
)
# Evaluate
evaluate_model(pipe, X_validation, y_validation, "RandomForestClassifier", "Actual Dta")
