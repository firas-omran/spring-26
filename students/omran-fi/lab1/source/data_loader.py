import kagglehub
import pandas as pd
import os

DATASET = "yasserh/titanic-dataset"
FILE = "Titanic-Dataset.csv"


def load_dataset():

    path = kagglehub.dataset_download(DATASET)

    file_path = os.path.join(path, FILE)

    df = pd.read_csv(file_path)

    return df