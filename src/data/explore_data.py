# -*- coding: utf-8 -*-

from random import sample
import numpy as np
import pandas as pd
from scipy.stats import zscore, normaltest, skew, kurtosis
from pathlib import Path
from IPython.display import display, Markdown
import logging

project_directory = Path(__file__).resolve().parents[2]
data_directory = project_directory / "data"
raw_data_directory = data_directory / "raw"
interim_data_directory = data_directory / "interim"


def list_datasets(folder="raw"):
    from pathlib import Path
    from IPython.display import display

    data_directory = Path(__file__).resolve().parents[2] / "data" / folder
    datasets_in_folder = tuple([path.name for path in data_directory.rglob("*.csv")])
    display(datasets_in_folder)


def overview_data(
    csv_file_name,
    display_mode="all",
    display_option="expanded",
    sample_mode="head",
    sample_size=10,
):
    display_option_validation = ("expanded", "condensed")
    sample_mode_validation = ("head", "random")

    if display_option not in display_option_validation:
        raise ValueError(f"display_option must be one of {display_option_validation}.")
    if sample_mode not in sample_mode_validation:
        raise ValueError(f"sample_mode must be one of {sample_mode_validation}.")

    path_to_csv_file = Path(raw_data_directory / csv_file_name)
    df = pd.read_csv(path_to_csv_file)
    pd.reset_option("display.max_columns")

    if (display_option == "expanded") & (
        len(df.columns) > pd.options.display.max_columns
    ):
        pd.set_option("display.max_columns", len(df.columns))

    display(Markdown("#### Dataset name (.csv)"))
    print(path_to_csv_file.name)

    if (
        (display_mode == "sample")
        or (display_mode == "all")
        or ("sample" in display_mode)
    ):
        if sample_mode == "head":
            display(Markdown("#### Dataset Records"), df.head(sample_size))
        else:
            display(
                Markdown("#### Dataset Random Samples"),
                df.sample(sample_size, random_state=42),
            )

    if (display_mode == "info") or (display_mode == "all") or ("info" in display_mode):
        display(Markdown("#### Dataset info"))
        df.info()

    if (
        (display_mode == "describe")
        or (display_mode == "all")
        or ("describe" in display_mode)
    ):
        display(
            Markdown("#### Dataset descriptive statistics"), df.describe(include="all")
        )

    pd.reset_option("display.max_columns")


def describe_features(dataset, display_option="expanded"):
    display_option_validation = ("expanded", "condensed")
    if display_option not in display_option_validation:
        raise ValueError(f"display_option must be one of {display_option_validation}.")

    path_to_csv_file = Path(raw_data_directory / "HomeCredit_columns_description.csv")
    df = pd.read_csv(path_to_csv_file, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"Table": "Dataset", "Row": "Column"}, inplace=True)
    pd.reset_option("display.max_rows")

    if dataset in ["application_train.csv", "application_test_student.csv"]:
        df_description_table = df[(df["Dataset"] == "application_{train|test}.csv")]
    else:
        df_description_table = df[(df["Dataset"] == dataset)]

    if display_option == "expanded":
        pd.set_option("display.max_rows", len(df_description_table))
        max_description_length = df_description_table["Description"].str.len().max()
        max_special_length = df_description_table["Special"].str.len().max()
        if max_special_length > pd.options.display.max_colwidth:
            with pd.option_context("display.max_colwidth", max_special_length + 1):
                display(df_description_table)
        elif max_description_length > pd.options.display.max_colwidth:
            with pd.option_context("display.max_colwidth", max_description_length + 1):
                display(df_description_table)
    else:
        display(df_description_table)

    pd.reset_option("display.max_rows")


def describe_feature(dataset, feature_or_column, display_option="expanded"):
    display_option_validation = ("expanded", "condensed")
    if display_option not in display_option_validation:
        raise ValueError(f"display_option must be one of {display_option_validation}.")

    path_to_csv_file = Path(raw_data_directory / "HomeCredit_columns_description.csv")
    df = pd.read_csv(path_to_csv_file, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"Table": "Dataset", "Row": "Column"}, inplace=True)

    if dataset in ["application_train.csv", "application_test_student.csv"]:
        df_description_instance = df[
            (df["Dataset"] == "application_{train|test}.csv")
            & (df["Column"] == feature_or_column)
        ]
    else:
        df_description_instance = df[
            (df["Dataset"] == dataset) & (df["Column"] == feature_or_column)
        ]

    if display_option == "expanded":
        description_length = df_description_instance["Description"].str.len()
        special_length = df_description_instance["Special"].str.len()
    if description_length > pd.options.display.max_colwidth:
        with pd.option_context("display.max_colwidth", description_length):
            display(df_description_instance)
    elif special_length > pd.options.display.max_colwidth:
        with pd.option_context("display.max_colwidth", special_length):
            display(df_description_instance)
    else:
        display(df_description_instance)


def create_dataframe(folder, csv_file_name):
    path_to_csv_file = Path(data_directory / folder / csv_file_name)
    df = pd.read_csv(path_to_csv_file)

    return df


def determine_type(df, feature):
    # Special Cases
    if feature == "SK_ID_CURR":
        return "id"
    elif feature == "TARGET":
        return "target"
    elif feature[:4] == "FLAG":
        return "categorical"
    # Standand Cases
    elif issubclass(df[feature].dtype.type, np.number):
        return "numeric"
    else:
        return "categorical"


def sum_quantile_outliers(df, feature, return_type):
    if issubclass(df[feature].dtype.type, np.float_):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_outliers = len(df[feature][Q1 - 1.5 * IQR >= df[feature]])
        upper_outliers = len(df[feature][df[feature] >= Q3 + 1.5 * IQR])

        if return_type == "lower":
            return int(lower_outliers)
        elif return_type == "upper":
            return int(upper_outliers)
        elif return_type == "both":
            return int(lower_outliers + upper_outliers)
    else:
        return " "


def sum_z_outliers(df, feature):
    if issubclass(df[feature].dtype.type, np.float_):
        zscores = zscore(df[feature]).abs()
        z_outliers_amount = len(zscores[zscores > 3])
        return int(z_outliers_amount)
    else:
        return " "


def test_normality(df, feature, alpha=0.05):
    if issubclass(df[feature].dtype.type, np.float_):
        _, pvalue = normaltest(df[feature])

        if pvalue < alpha:  # Null hypothesis: distribution is normal
            return "no"
        else:
            return "yes"
    else:
        return " "


def calculate_skewness(df, feature):
    if issubclass(df[feature].dtype.type, np.float_):
        return round(kurtosis(df[feature], nan_policy="omit"), 2)
    else:
        return " "


def calculate_kurtosis(df, feature):
    """ """
    if issubclass(df[feature].dtype.type, np.float_):
        return np.round(skew(df[feature], nan_policy="omit"), 2)
    else:
        return " "


def calculate_correlation(df, feature):
    if issubclass(df[feature].dtype.type, np.number):
        return round(df[feature].corr(df["TARGET"]), 2)
    else:
        return " "


def create_exploratory_dataset(dataset_for_analysis="application_train.csv"):
    df = create_dataframe("raw", dataset_for_analysis)

    df_exploratory = pd.read_csv(
        raw_data_directory / "HomeCredit_columns_description.csv", index_col=0, encoding_errors="ignore"
    )
    df_exploratory.reset_index(drop=True, inplace=True)
    df_exploratory.rename(columns={"Table": "Dataset", "Row": "Column"}, inplace=True)
    df_exploratory = df_exploratory.replace(
        {
            "application_{train|test}.csv": "application_train.csv",
            "SK_BUREAU_ID": "SK_ID_BUREAU",
        }
    )
    df_exploratory = df_exploratory[df_exploratory["Dataset"] == dataset_for_analysis]
    df_exploratory.drop(labels=["Dataset", "Special"], axis=1, inplace=True)

    df_exploratory["NanPercentage"] = (
        df_exploratory["Column"].map(df.isnull().mean().round(4))
    ) * 100
    df_exploratory["DataType"] = df_exploratory["Column"].map(df.dtypes)
    df_exploratory["FeatureType"] = df_exploratory["Column"].map(
        lambda column: determine_type(df, column)
    )
    df_exploratory["Q1Outliers"] = df_exploratory["Column"].map(
        lambda column: sum_quantile_outliers(df, column, return_type="lower")
    )
    df_exploratory["Q3Outliers"] = df_exploratory["Column"].map(
        lambda column: sum_quantile_outliers(df, column, return_type="upper")
    )
    df_exploratory["Z3Outliers"] = df_exploratory["Column"].map(
        lambda column: sum_z_outliers(df, column)
    )
    df_exploratory["NormalDistribution"] = df_exploratory["Column"].map(
        lambda column: test_normality(df, column)
    )
    df_exploratory["Skewness"] = df_exploratory["Column"].map(
        lambda column: calculate_skewness(df, column)
    )
    df_exploratory["Kurtosis"] = df_exploratory["Column"].map(
        lambda column: calculate_kurtosis(df, column)
    )
    df_exploratory["CorrWithTarget"] = df_exploratory["Column"].map(
        lambda column: calculate_correlation(df, column)
    )

    df_exploratory.to_csv(
        interim_data_directory / "application_exploratory.csv", index=False
    )


def create_decision_dataset():
    if not (interim_data_directory / "application_exploratory.csv").is_file():
        create_exploratory_dataset()
    df_decision = create_dataframe("interim", "application_exploratory.csv")

    # Nan Decision
    conditions = (
        df_decision["NanPercentage"] > 30,
        (df_decision["NanPercentage"] > 0) & (df_decision["DataType"] == "object"),
        (df_decision["NanPercentage"] > 0) & (df_decision["DataType"] != "object"),
    )
    values = ("drop", "frequent", "median")
    df_decision["NanDecision"] = np.select(conditions, values, default=" ")
    ## Exceptions
    df_decision.loc[
        df_decision["Column"] == "CODE_GENDER", ["NanDecision"]
    ] = "frequent"
    df_decision.loc[
        df_decision["Column"] == "NAME_FAMILY_STATUS", ["NanDecision"]
    ] = "frequent"
    df_decision.loc[
        df_decision["Column"] == "ORGANIZATION_TYPE", ["NanDecision"]
    ] = "random"

    # DataType Decision
    df_decision["TypeDecision"] = np.where(
        (df_decision["DataType"] == "object") & (df_decision["NanDecision"] != "drop"),
        "OrdinalEncoder",
        " ",
    )

    # Outliers Decision (arbitrary)
    cols_with_high_outliers = (
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "REGION_POPULATION_RELATIVE",
        "DAYS_REGISTRATION",
    )  # Z-score higher than 3
    df_decision["OutliersDecision"] = np.where(
        df_decision["Column"].isin(cols_with_high_outliers), "remove", " "
    )

    # Correlation Decision
    df_decision["CorrDecision"] = " "
    df_decision["CorrDecision"] = np.where(
        df_decision["Column"].str.startswith("FLAG_DOCUMENT_"), "group", " "
    )
    df_decision.loc[df_decision["Column"] == "SK_ID_CURR", ["CorrDecision"]] = "drop"

    # New Features
    new_features = pd.DataFrame(
        {
            "Column": [
                "AMT_CREDIT_div_AMT_ANNUITY",
                "AMT_INCOME_TOTAL_div_AMT_CREDIT",
                "AMT_INCOME_TOTAL_div_CNT_FAM_MEMBERS",
            ],
            "TypeDecision": 3 * ["create"],
        }
    )
    df_decision = pd.concat((df_decision, new_features), ignore_index=True)
    df_decision.fillna(value=" ", inplace=True)

    # Extras
    cols_with_hidden_nans = ("CODE_GENDER", "NAME_FAMILY_STATUS", "ORGANIZATION_TYPE")
    df_decision["Extras"] = np.where(
        df_decision["Column"].isin(cols_with_hidden_nans), "identify hidden nans", " "
    )

    # Removing columns from df_exploratory
    df_decision = df_decision.loc[
        :,
        (
            "Column",
            "NanDecision",
            "TypeDecision",
            "OutliersDecision",
            "CorrDecision",
            "Extras",
        ),
    ]

    # Saving DataFrame
    df_decision.to_csv(interim_data_directory / "application_decision.csv", index=False)


def main():
    create_exploratory_dataset()
    logging.info("Exploratory Dataset is now available on /data/interim folder")
    create_decision_dataset()
    logging.info("Decision Dataset is now available on /data/interim folder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
