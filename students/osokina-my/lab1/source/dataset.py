import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_adult_dataset(
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
) -> pd.DataFrame:
    """
    Загрузка датасета Adult (income >50K предсказание).
    Содержит пропуски в workclass и occupation (значение '?').
    """
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    df = pd.read_csv(
        url, header=None, names=columns,
        sep=r",\s*", engine="python", na_values="?"
    )
    # Удаление пробелов в строковых столбцах
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df


def introduce_missing_values(
    df: pd.DataFrame,
    fraction: float = 0.05,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Добавляет 5–10% пропусков в каждый признак.
    Используется, если в датасете нет пропусков.
    """
    rng = np.random.default_rng(random_state)
    result = df.copy()
    for col in result.columns:
        if col == "income" or col == "target":
            continue
        n_missing = int(len(result) * fraction)
        idx = rng.choice(len(result), size=n_missing, replace=False)
        result.loc[result.index[idx], col] = np.nan
    return result


def get_feature_info(df: pd.DataFrame, target_col: str = "income") -> Tuple[list, list]:
    """
    Разделяет признаки на категориальные и количественные.
    """
    X = df.drop(columns=[target_col])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols
