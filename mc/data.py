from typing import Iterable

import numpy as np
import pandas as pd
from catboost import Pool
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalizer
from sklearn.model_selection import GroupShuffleSplit

from .featurize import featurize
from .utils import disable_rdkit_log


def split_into_phases(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into three phases:
    1. phase without restrictions on the vote value
    2. active learning phase with similar molecules
    3. all the votes must be unique

    Parameters:
    -----------
    param: raw_data: data with features and metadata
    param: indices_to_pick: indices of users to pick (alab members)

    Returns:
    -----------
    tuple, (phase1, phase2, phase3) of three Pandas DataFrames representing the three phases
    """

    time1 = "2022-11-06 00:00:00.000000"
    time2_start, time2_finish = (
        "2022-12-05 18:49:30.986130",
        "2023-03-14 00:00:00.000000",
    )

    phase1 = raw_data[raw_data["vote_date"] <= time1]
    phase2 = raw_data[
        (time2_start < raw_data["vote_date"]) & (raw_data["vote_date"] < time2_finish)
    ]
    phase3 = raw_data[raw_data["vote_date"] > time2_finish]

    return phase1, phase2, phase3


def data2pool(
    df: pd.DataFrame,
    weights: Iterable[float] | None = None,
) -> Pool:
    """
    Transform the data into a Pool object for CatBoost according
    to the given weights for each phase and indices to pick (alab members)

    Parameters:
    -----------
    param: df: data with features and metadata
    param: indices_to_pick: indices of users to pick (alab members)
    param: weights: weights for each phase

    Returns:
    -----------
    Pool object for CatBoost
    """
    disable_rdkit_log()
    cols_to_drop = ["user_id", "vote_date", "smiles", "vote", "mol", "quids"]
    if weights is not None:
        assert len(weights) == 3, (
            f"weights must be of length 3, got {len(weights)} parameters"
        )

        phase1, phase2, phase3 = split_into_phases(df)
        df = pd.concat([phase1, phase2, phase3])
        p1, p2, p3 = weights
        weights = np.concatenate(
            [
                np.full(len(phase1), p1),
                np.full(len(phase2), p2),
                np.full(len(phase3), p3),
            ]
        )

    X, y, quids = df.drop(columns=cols_to_drop), df.vote, df.quids

    return Pool(X, y, group_id=quids, group_weight=weights)


def custom_split(df, test_size=0.1):
    """
    split the dataset into train and test using quids as a group identifier

    :param df: processed with `transform_df` function data
    :param test_size: float, default=.20
    :param n_splits: int, default=5
    :param random_state: int, default=42
    :param smi: bool, default=False
    :param quids: bool, default=False
    :return: split of the dataset into train and test sets with features, labels and (optionally) quids
    """
    gss = GroupShuffleSplit(test_size=test_size, n_splits=5, random_state=42).split(
        df, groups=df["quids"]
    )

    X_train_inds, X_test_inds = next(gss)
    train, test = df.iloc[X_train_inds], df.iloc[X_test_inds]

    return train, test


def build_pools(df, weights, split_by_phase=False):
    """
    map data to Pool objects for CatBoost model

    :param df: processed with `transform_df` function data
    :return: tuple, (train_pool, test_pool) of two Pool objects for CatBoost model
    """
    train, test = custom_split(df)
    train_pool = data2pool(train, weights)

    if split_by_phase:
        p1_test, p2_test, p3_test = split_into_phases(test)
        return train_pool, *map(data2pool, [p1_test, p2_test, p3_test])

    test_pool = data2pool(test, None)
    return train_pool, test_pool


def build_inference_df(smidf) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    build the dataset for prediction and smiles string

    :param data: Pandas DataFrame with one column 'smiles'
    :return: tuple, (smi_df, smi) of two DataFrames
    """

    normalizer = Normalizer()
    smidf["mol"] = smidf["smiles"].apply(lambda x: Chem.AddHs(Chem.MolFromSmiles(x)))
    smidf["mol"] = smidf["mol"].apply(lambda x: normalizer.normalize(x))
    smidf["smiles"] = smidf["mol"].apply(lambda x: Chem.MolToSmiles(x))
    smidf = featurize(smidf)
    smi_df, smi = smidf.loc[:, ~smidf.columns.isin(["smiles", "mol"])], smidf["smiles"]
    return smi_df, smi
