import pandas as pd
import os
from typing import Tuple, Optional

# the default folder of the processed data
PROCESSED_FOLDER = "/data/processed/benchmark/inhospitalmortality"

# The names of the features as generated for Logistic Regression by Harutyunyan et al. (2019)
COLUMN_NAMES = [str(i) for i in range(714)]

# A dictionary mapping an OOD group name to its corresponding (column, column_value) pair
OOD_MAPPINGS = {'Emergency/Urgent admissions': ('ADMISSION_TYPE', 'EMERGENCY'),
                'Elective admissions': ('ADMISSION_TYPE', 'ELECTIVE'),
                'Ethnicity: Asian': ('Ethnicity', 1),
                'Ethnicity: Black/African American': ('Ethnicity', 2),
                'Ethnicity: Hispanic/Latino': ('Ethnicity', 3),
                'Ethnicity: White': ('Ethnicity', 4),
                'Female': ('GENDER', 'F'),
                'Male': ('GENDER', 'M'),
                'Thyroid disorders': ('Thyroid disorders', True),
                'Acute and unspecified renal failure': (
                    'Acute and unspecified renal failure', True),
                'Pancreatic disorders (not diabetes)': (
                    'Pancreatic disorders (not diabetes)', True),
                'Epilepsy; convulsions': ('Epilepsy; convulsions', True),
                'Hypertension with complications and secondary hypertension': (
                    'Hypertension with complications and secondary hypertension', True)}


def load_data(dir_name: str = PROCESSED_FOLDER) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from directory.

    Parameters
    ----------
    dir_name: str
        The directory with the data, default "/data/processed/benchmark/inhospitalmortality"

    Returns
    -------
    train_data: pd.DataFrame
        The training data.
    val_data: pd.DataFrame
        The validation data.
    test_data: pd.DataFrame
        The test data.

   """
    train_data = pd.read_csv(os.path.join(dir_name, 'train_data_processed_w_static.csv'),
                             index_col=0)
    val_data = pd.read_csv(os.path.join(dir_name, 'val_data_processed_w_static.csv'),
                           index_col=0)
    test_data = pd.read_csv(os.path.join(dir_name, 'test_data_processed_w_static.csv'),
                            index_col=0)
    return train_data, val_data, test_data


def split_by_ood_name(df: pd.DataFrame, ood_name: str, ood_value: Optional[int, str, bool]) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by OOD column name and corresponding OOD value.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to split.
    ood_name: str
        The OOD column name
    ood_value: str


    Returns
    -------
    ood_df : pd.DataFrame
        The part of the dataframe with the OOD value.
    non_ood_df: pd.DataFrame
        The part of the dataframe without the OOD value.
    """
    ood_df = df[df[ood_name] == ood_value]
    non_ood_df = df[~(df[ood_name] == ood_value)]
    return ood_df, non_ood_df
