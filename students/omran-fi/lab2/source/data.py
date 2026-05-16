from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET = "yasserh/titanic-dataset"
DATA_FILE = "Titanic-Dataset.csv"
TARGET = "Survived"

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "FarePerPerson",
    "TicketGroupSize",
]
CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "IsAlone",
]


def load_raw_titanic(project_root: Path) -> pd.DataFrame:
    """Load Titanic data, using a local cached copy when available."""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    local_path = data_dir / "titanic.csv"

    if local_path.exists():
        return pd.read_csv(local_path)

    downloaded = Path(kagglehub.dataset_download(DATASET)) / DATA_FILE
    df = pd.read_csv(downloaded)
    df.to_csv(local_path, index=False)
    return df


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"
    }
    df["Title"] = df["Title"].replace(list(rare_titles), "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, "Yes", "No")
    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts).astype(float)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
    return df


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(project_root: Path, test_size: float = 0.2, random_state: int = 42):
    raw = add_titanic_features(load_raw_titanic(project_root))
    X = raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = raw[TARGET].astype(int).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    X_train = pd.DataFrame(X_train_arr, columns=feature_names).reset_index(drop=True)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    dataset_info = {
        "n_samples": int(len(raw)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(y.mean()),
        "raw_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "prepared_features": feature_names,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_values": {k: int(v) for k, v in raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum().items()},
    }
    return X_train, X_test, y_train, y_test, dataset_info
