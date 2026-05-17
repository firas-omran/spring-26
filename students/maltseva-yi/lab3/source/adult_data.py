import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_adult_data():
    # Загрузка Adult dataset из UCI репозитория.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']
    df = pd.read_csv(url, names=columns, sep=r',\s*', na_values='?', engine='python')
    df['income'] = (df['income'] == '>50K').astype(int)
    df.dropna(subset=['income'], inplace=True)
    X = df.drop('income', axis=1)
    y = df['income'].values
    return X, y

def get_preprocessor():
    # Создаёт ColumnTransformer для кодирования и заполнения пропусков.
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                        'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race', 'sex',
                            'native-country']

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor, numeric_features, categorical_features