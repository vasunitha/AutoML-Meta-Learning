from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np


def build_preprocessor(X):
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )
    return preprocessor


def get_candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "lgbm": LGBMClassifier(random_state=42),
        "catboost": CatBoostClassifier(verbose=0, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }


def build_pipelines(X):
    preprocessor = build_preprocessor(X)
    models = get_candidate_models()

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    return pipelines
