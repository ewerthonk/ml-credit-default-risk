# Basic Imports
from src.data.explore_data import create_dataframe
import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging

# Sklearn Base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

# Sklearn Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Imblearn
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

# Feature Engine
from feature_engine.imputation import (
    MeanMedianImputer,
    CategoricalImputer,
    RandomSampleImputer,
)
from feature_engine.encoding import OrdinalEncoder
from feature_engine.outliers import OutlierTrimmer
from feature_engine.creation import MathFeatures, RelativeFeatures
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper


# Defining paths
project_directory = Path(__file__).resolve().parents[2]
data_directory = project_directory / "data"
raw_data_directory = data_directory / "raw"
interim_data_directory = data_directory / "interim"
processed_data_directory = data_directory / "processed"


class NanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nan_string, variables):
        self.nan_string = nan_string
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for variable in self.variables:
            X.loc[X[variable] == self.nan_string, variable] = np.nan
        return X


def trim_outliers(
    X,
    y,
    capping_method="gaussian",
    tail="right",
    fold=3,
    variables=None,
    missing_values="raise",
):
    trimmer = OutlierTrimmer(
        capping_method=capping_method,
        tail=tail,
        fold=fold,
        variables=variables,
        missing_values=missing_values,
    )
    trimmer.fit(X)
    X_new = trimmer.transform(X)
    y_new = y.loc[X_new.index]

    return X_new, y_new


# Loading DataFrames
df = create_dataframe("raw", "application_train.csv")
df_decision = create_dataframe("interim", "application_decision.csv")

# Creating X and y
X = df.drop(labels=("TARGET"), axis=1).copy()
y = df["TARGET"].copy()

# Splitting into Train and Validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Defining features to be processed in the pipeline
drop_cols = df_decision["Column"][df_decision.isin(["drop"]).any(axis=1)].to_list()
xna_cols = ["CODE_GENDER", "ORGANIZATION_TYPE"]
unknown_cols = ["NAME_FAMILY_STATUS"]
ordinal_encoder_cols = df_decision["Column"][
    df_decision["TypeDecision"].str.contains("OrdinalEncoder")
].to_list()
nan_to_frequent_cols = df_decision["Column"][
    df_decision["NanDecision"] == "frequent"
].to_list()
nan_to_random_cols = df_decision["Column"][
    df_decision["NanDecision"] == "random"
].to_list()
outliers_cols = df_decision["Column"][
    df_decision["OutliersDecision"] == "remove"
].to_list()
group_cols = df_decision["Column"][df_decision["CorrDecision"] == "group"].to_list()
rate_annuity_cols = ["AMT_CREDIT", "AMT_ANNUITY"]
rate_credit_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT"]
rate_income_cols = ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]

# Creating Pipeline
preprocessing_pipeline = imbpipeline(
    [
        # Step 1. Drop Columns with NanPercentage > 30% and SK_ID_CURR
        ("drop_features", DropFeatures(features_to_drop=drop_cols)),
        # Step 2. Reveal Hidden NaNs
        ("xna_transformer", NanTransformer(nan_string="XNA", variables=xna_cols)),
        (
            "unknown_transformer",
            NanTransformer(nan_string="Unknown", variables=unknown_cols),
        ),
        # Step 3. Impute NaN Values
        ("num_imputer", MeanMedianImputer()),
        (
            "cat1_imputer",
            CategoricalImputer(
                imputation_method="frequent", variables=nan_to_frequent_cols
            ),
        ),
        (
            "cat2_imputer",
            RandomSampleImputer(random_state=42, variables=nan_to_random_cols),
        ),
        # Step 4. Encode Categorical Features
        ("ordinal_encoder", OrdinalEncoder(variables=ordinal_encoder_cols)),
        # Step 5. Trim Outliers (with Z-score > +-3)
        (
            "trim_outliers",
            FunctionSampler(
                func=trim_outliers,
                validate=False,
                kw_args={
                    "capping_method": "gaussian",
                    "tail": "both",
                    "variables": outliers_cols,
                },
            ),
        ),
        # Step 6. Group Document Columns
        (
            "grouper",
            MathFeatures(
                variables=group_cols,
                func=[np.sum],
                new_variables_names=["DOCUMENTS_PROVIDED"],
                drop_original=True,
            ),
        ),
        # Step 7. Create new Features
        (
            "AMT_CREDIT_div_AMT_ANNUITY_creator",
            RelativeFeatures(
                variables=[rate_annuity_cols[0]],
                reference=[rate_annuity_cols[1]],
                func=["div"],
                drop_original=False,
            ),
        ),
        (
            "AMT_INCOME_TOTAL_div_AMT_CREDIT_creator",
            RelativeFeatures(
                variables=[rate_credit_cols[0]],
                reference=[rate_credit_cols[1]],
                func=["div"],
                drop_original=False,
            ),
        ),
        (
            "AMT_INCOME_TOTAL_div_CNT_FAM_MEMBERS_creator",
            RelativeFeatures(
                variables=[rate_income_cols[0]],
                reference=[rate_income_cols[1]],
                func=["div"],
                drop_original=False,
            ),
        ),
        # Step 8. Scale features
        ("scaler", SklearnTransformerWrapper(transformer=MinMaxScaler())),
    ]
)


def run_preprocessing_pipeline(
    preprocessing_pipeline=preprocessing_pipeline,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
):
    # Through the Pipeline
    X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_valid_transformed = preprocessing_pipeline.transform(X_valid)

    # Saving DataFrames
    X_train.to_csv(processed_data_directory / "X_train.csv", index=False)
    y_train.to_csv(processed_data_directory / "y_train.csv", index=False)
    X_valid_transformed.to_csv(
        processed_data_directory / "X_valid_transformed.csv", index=False
    )
    y_valid.to_csv(processed_data_directory / "y_valid.csv", index=False)


def create_classifiers_dataset(
    preprocessing_pipeline=preprocessing_pipeline, X_train=X_train, y_train=y_train
):
    classifiers_pipeline = clone(preprocessing_pipeline)
    classifiers_pipeline.steps.append(
        ("classifier", DummyClassifier(strategy="stratified", random_state=42))
    )

    classifiers = [
        {"classifier": (DummyClassifier(random_state=42),)},
        {"classifier": (LogisticRegression(random_state=42, max_iter=1000),)},
        {"classifier": (KNeighborsClassifier(),)},
        {"classifier": (DecisionTreeClassifier(random_state=42),)},
        {"classifier": (LinearSVC(random_state=42),)},
        {"classifier": (RandomForestClassifier(random_state=42),)},
        {"classifier": (AdaBoostClassifier(random_state=42),)},
        {"classifier": (GradientBoostingClassifier(random_state=42),)},
        {"classifier": (XGBClassifier(random_state=42),)},
        {"classifier": (CatBoostClassifier(random_state=42, verbose=False),)},
        {"classifier": (LGBMClassifier(random_state=42),)},
    ]

    classifiers_grid = GridSearchCV(
        estimator=classifiers_pipeline,
        param_grid=classifiers,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    classifiers_grid.fit(X_train, y_train)

    df_classifiers = pd.DataFrame(classifiers_grid.cv_results_)

    # Customizing Classifiers DataFrame
    df_classifiers.drop(
        labels=(
            ["std_fit_time", "std_score_time"]
            + [col for col in df_classifiers.columns.to_list() if "split" in col]
        ),
        axis=1,
        inplace=True,
    )
    df_classifiers.loc[:, ["params"]] = "Default"
    df_classifiers.rename(columns={"param_classifier": "classifiers"}, inplace=True)
    df_classifiers.sort_values(by=["rank_test_score"], inplace=True)
    df_classifiers = df_classifiers[
        [
            "rank_test_score",
            "classifiers",
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "mean_score_time",
        ]
    ]
    regex = r"(^.+?(?=\())|(\b\w+Classifier)"
    df_classifiers["classifiers"] = df_classifiers["classifiers"].map(
        lambda x: re.search(regex, str(x)).group()
    )

    # Saving Dataframe
    df_classifiers.to_csv(interim_data_directory / "classifiers.csv", index=False)


def main():
    logging.info("Running Pre-processing Pipeline")
    run_preprocessing_pipeline()
    logging.info("Pre-processed Datasets are now available /data/processed folder")
    logging.info("Creating Classifiers Dataset")
    create_classifiers_dataset()
    logging.info("Classifiers Dataset is now available on /data/interim folder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
