# Basic Imports
from src.features.pipeline import (
    preprocessing_pipeline,
    X_train,
    y_train,
)
from src.data.explore_data import create_dataframe

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Sklearn Base
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone

# Sklearn Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# Models Persistence
from joblib import dump, load

# Paths
# Defining paths
project_directory = Path(__file__).resolve().parents[2]
models_directory = project_directory / "models"
data_directory = project_directory / "data"
processed_data_directory = data_directory / "processed"

# Defining Cross-Validation parameters
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def create_catboost_model(preprocessing_pipeline=preprocessing_pipeline):
    catboost_pipeline = clone(preprocessing_pipeline)
    catboost_pipeline.steps.append(
        ("classifier", CatBoostClassifier(random_state=42, verbose=False))
    )
    catboost_params = {
        "classifier__iterations": np.linspace(10, 1000, 5, dtype=int),
        "classifier__learning_rate": np.linspace(0.01, 0.8, 5, dtype=float),
        "classifier__depth": np.linspace(6, 10, 5, dtype=int),
    }
    catboost = RandomizedSearchCV(
        estimator=catboost_pipeline,
        param_distributions=catboost_params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    catboost.fit(X_train, y_train)
    dump(catboost, (models_directory / "catboost.joblib"))


def create_lgbm_model(preprocessing_pipeline=preprocessing_pipeline):
    lgbm_pipeline = clone(preprocessing_pipeline)
    lgbm_pipeline.steps.append(("classifier", LGBMClassifier(random_state=42)))
    lgbm_params = {
        "classifier__learning_rate": np.linspace(0.01, 0.8, 5, dtype=float),
        "classifier__max_depth": np.linspace(2, 16, 5, dtype=int),
        "classifier__n_estimators": np.linspace(100, 6000, 5, dtype=int),
        "classifier__num_leaves": np.linspace(20, 3000, 5, dtype=int),
    }
    lgbm = RandomizedSearchCV(
        estimator=lgbm_pipeline,
        param_distributions=lgbm_params,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    lgbm.fit(X_train, y_train)
    dump(lgbm, (models_directory / "lgbm.joblib"))


def create_final_model(preprocessing_pipeline=preprocessing_pipeline):
    model = clone(preprocessing_pipeline)
    model.steps.append(
        (
            "classifier",
            VotingClassifier(
                estimators=(
                    (
                        "LightGBM",
                        load(models_directory / "lgbm.joblib").best_estimator_[-1],
                    ),
                    (
                        "CatBoost",
                        load(models_directory / "catboost.joblib").best_estimator_[-1],
                    ),
                ),
                voting="soft",
            ),
        )
    )
    model.fit(X_train, y_train)
    dump(model, (models_directory / "final_model.joblib"))


def create_submission():
    X_test = create_dataframe("raw", "application_test_student.csv")
    predict_probas = load(
        processed_data_directory / "final_model.joblib"
    ).predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {"SK_ID_CURR": X_test["SK_ID_CURR"], "TARGET": predict_probas}
    )
    submission.to_csv(processed_data_directory / "submission.csv", index=False)


def main():
    logging.info("Fitting CatBoost Model")
    create_catboost_model()
    logging.info("CatBoost Model created, fitted and saved into /models folder")
    logging.info("Fitting LightGBM Model")
    create_lgbm_model()
    logging.info("LightGBM Model created, fitted and saved into /models folder")
    logging.info("Fitting Final Model")
    create_final_model()
    logging.info("Final Model created, fitted and saved into /models folder")
    logging.info("Creating Test Submission")
    create_submission()
    logging.info("Test Submission .csv created and saved into /data/processed folder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
