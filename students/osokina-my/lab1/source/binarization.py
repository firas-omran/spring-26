import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class Binarizer:
    """
    Бинаризация признаков:
    - Категориальные: каждое значение -> отдельный бинарный признак (one-hot)
    - Количественные: по медиане (<= медиана, > медиана)
    """

    def __init__(self):
        self.thresholds_: Dict[str, float] = {}
        self.categories_: Dict[str, List[Any]] = {}
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> "Binarizer":
        self.feature_types_ = {}
        self.thresholds_ = {}
        self.categories_ = {}
        self.feature_names_ = []

        for col in cat_cols:
            vals = X[col].dropna().unique().tolist()
            self.categories_[col] = sorted([v for v in vals if pd.notna(v)])
            self.feature_types_[col] = "categorical"
            for v in self.categories_[col]:
                self.feature_names_.append(f"{col}={v}")

        for col in num_cols:
            med = X[col].median()
            self.thresholds_[col] = med
            self.feature_types_[col] = "numeric"
            self.feature_names_.append(f"{col}<={med:.2f}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for i in range(len(X)):
            row = {}
            for col, vals in self.categories_.items():
                v = X.iloc[i][col]
                if pd.isna(v):
                    for cat in vals:
                        row[f"{col}={cat}"] = np.nan
                else:
                    for cat in vals:
                        row[f"{col}={cat}"] = 1 if v == cat else 0
            for col, thresh in self.thresholds_.items():
                val = X[col].iloc[i]
                if pd.isna(val):
                    row[f"{col}<={thresh:.2f}"] = np.nan
                else:
                    row[f"{col}<={thresh:.2f}"] = 1 if val <= thresh else 0
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names_)

    def fit_transform(
        self, X: pd.DataFrame, cat_cols: List[str], num_cols: List[str]
    ) -> pd.DataFrame:
        self.fit(X, cat_cols, num_cols)
        return self.transform(X)
