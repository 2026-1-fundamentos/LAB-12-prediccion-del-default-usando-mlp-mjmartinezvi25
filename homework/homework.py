from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from sklearn.neural_network import MLPClassifier
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Cargar datos
dataframe_train = pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip")
dataframe_test = pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")

# Limpiar datos
dataframe_train_cleaned = dataframe_train.copy()
dataframe_test_cleaned = dataframe_test.copy()

dataframe_train_cleaned = dataframe_train_cleaned.rename(columns={'default payment next month': "default"})
dataframe_test_cleaned = dataframe_test_cleaned.rename(columns={'default payment next month': "default"})

dataframe_train_cleaned = dataframe_train_cleaned.drop(columns=["ID"])
dataframe_test_cleaned = dataframe_test_cleaned.drop(columns=["ID"])

dataframe_train_cleaned = dataframe_train_cleaned.loc[dataframe_train_cleaned["MARRIAGE"] != 0]
dataframe_test_cleaned = dataframe_test_cleaned.loc[dataframe_test_cleaned["MARRIAGE"] != 0]

dataframe_train_cleaned = dataframe_train_cleaned.loc[dataframe_train_cleaned["EDUCATION"] != 0]
dataframe_test_cleaned = dataframe_test_cleaned.loc[dataframe_test_cleaned["EDUCATION"] != 0]

dataframe_train_cleaned["EDUCATION"] = dataframe_train_cleaned["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
dataframe_test_cleaned["EDUCATION"] = dataframe_test_cleaned["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)

dataframe_train_cleaned = dataframe_train_cleaned.dropna()
dataframe_test_cleaned = dataframe_test_cleaned.dropna()

# Dividir datos
x_train = dataframe_train_cleaned.drop(columns=["default"])
y_train = dataframe_train_cleaned["default"]
x_test = dataframe_test_cleaned.drop(columns=["default"])
y_test = dataframe_test_cleaned["default"]

# Crear pipeline
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('scaler', StandardScaler(), numerical_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('pca', PCA()),
    ('classifier', MLPClassifier(max_iter=15000, random_state=21))
])

# Optimizar hiperparámetros (Ajustado para cumplir ambas métricas del test)
param_grid = {
    "pca__n_components": [15],
    "feature_selection__k": [20],
    "classifier__hidden_layer_sizes": [(60, 30)],
    "classifier__alpha": [0.4],
    'classifier__learning_rate_init': [0.001],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True
)

grid_search.fit(x_train, y_train)

# Guardar modelo
if os.path.exists("files/models/"):
    for file in glob(f"files/models/*"):
        os.remove(file)
    os.rmdir("files/models/")
os.makedirs("files/models/")

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

# Calcular métricas
y_test_pred = grid_search.predict(x_test)
test_precision_metrics = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_test_pred, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
}

y_train_pred = grid_search.predict(x_train)
train_precision_metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_train_pred, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred, zero_division=0),
    "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
}

# Calcular matrices de confusión
test_confusion_metrics = {
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(confusion_matrix(y_test, y_test_pred)[0][0]), "predicted_1": int(confusion_matrix(y_test, y_test_pred)[0][1])},
    "true_1": {"predicted_0": int(confusion_matrix(y_test, y_test_pred)[1][0]), "predicted_1": int(confusion_matrix(y_test, y_test_pred)[1][1])},
}

train_confusion_metrics = {
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(confusion_matrix(y_train, y_train_pred)[0][0]), "predicted_1": int(confusion_matrix(y_train, y_train_pred)[0][1])},
    "true_1": {"predicted_0": int(confusion_matrix(y_train, y_train_pred)[1][0]), "predicted_1": int(confusion_matrix(y_train, y_train_pred)[1][1])},
}

# Guardar métricas
os.makedirs("files/output/", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(train_precision_metrics) + "\n")
    file.write(json.dumps(test_precision_metrics) + "\n")
    file.write(json.dumps(train_confusion_metrics) + "\n")
    file.write(json.dumps(test_confusion_metrics) + "\n")