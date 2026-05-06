from pathlib import Path
import zipfile

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import requests


# Базовые пути проекта и архива с датасетом.
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
ARCHIVE_NAME = "diabetes-prediction-dataset.zip"
DATASET_FILE_NAME = "diabetes_prediction_dataset.csv"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "iammustafatz/diabetes-prediction-dataset"
)


def load_archive() -> None:
    """
    Загружает архив с датасетом, если он отсутствует локально.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Архив сохраняется в директорию data.

    Fallbacks:
        Если архив уже существует, загрузка пропускается.
    """
    # Создаем директорию с данными перед сохранением архива.
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DATA_PATH / ARCHIVE_NAME

    # Не загружаем файл повторно, если он уже есть на диске.
    if output_file.exists():
        return

    # Получаем архив потоково, чтобы не держать весь файл в памяти.
    response = requests.get(DATASET_URL, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()

    # Сохраняем архив порциями фиксированного размера.
    with output_file.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                archive_file.write(chunk)


def load_df() -> pd.DataFrame:
    """
    Читает CSV-файл с датасетом из локального ZIP-архива.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        pandas.DataFrame: Таблица с признаками и целевой переменной.

    Fallbacks:
        Если архив отсутствует или поврежден, исключение передается вызывающему коду.
    """
    # Открываем архив и читаем CSV без промежуточной распаковки на диск.
    archive_path = DATA_PATH / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open(DATASET_FILE_NAME) as dataset_file:
            dataframe = pd.read_csv(dataset_file)

    return dataframe


def is_categorical_feature(values: pd.Series) -> bool:
    """
    Определяет, содержит ли столбец нечисловые значения.

    Parameters:
        values (pandas.Series): Значения одного признака. По умолчанию: нет.

    Returns:
        bool: True, если столбец нужно считать категориальным.

    Fallbacks:
        Если все непустые значения приводятся к числу, столбец считается числовым.
    """
    # Пробуем привести непустые значения к числу, чтобы отделить категории от чисел.
    non_null_values = values.dropna()
    numeric_values = pd.to_numeric(non_null_values, errors="coerce")

    # Наличие пропусков после приведения означает, что в столбце есть строки-категории.
    return numeric_values.isna().any()


def balance_train_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names,
    encoder,
    save_categorical: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Балансирует обучающую выборку с помощью oversampling и undersampling.

    Parameters:
        features (numpy.ndarray): Обучающие признаки. По умолчанию: нет.
        labels (numpy.ndarray): Метки обучающей выборки. По умолчанию: нет.
        feature_names: Названия признаков. По умолчанию: нет.
        dataframe (pandas.DataFrame): Исходный датафрейм для определения типов признаков. По умолчанию: нет.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Сбалансированные признаки и метки.

    Fallbacks:
        Если категориальных признаков нет, категориальное кодирование не применяется.
    """
    # Собираем признаки в DataFrame, чтобы кодирование шло по именам столбцов.
    features_dataframe = pd.DataFrame(features, columns=feature_names).copy()

    # Определяем индексы и имена категориальных признаков для корректного oversampling.
    categorical_feature_names = []
    for feature_name in feature_names:
        # Проверяем столбец по фактическим значениям обучающей выборки, а не только по dtype.
        if is_categorical_feature(features_dataframe[feature_name]):
            categorical_feature_names.append(feature_name)

    categorical_indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if feature_name in categorical_feature_names
    ]

    # Кодируем только категориальные признаки, чтобы вернуть их после ресемплинга.
    if categorical_feature_names:
        features_dataframe[categorical_feature_names] = encoder.fit_transform(
            features_dataframe[categorical_feature_names]
        )

    # Приводим всю матрицу к float для работы samplers.
    encoded_features = features_dataframe.to_numpy(dtype=float)

    # Увеличиваем долю миноритарного класса до 0.1 от мажоритарного.
    oversampler = (
        SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=0.1,
            random_state=42,
        )
        if categorical_indices
        else SMOTE(
            sampling_strategy=0.1,
            random_state=42,
        )
    )
    oversampled_features, oversampled_labels = oversampler.fit_resample(
        encoded_features,
        labels,
    )

    # Уменьшаем мажоритарный класс до отношения 0.5 после oversampling.
    undersampler = RandomUnderSampler(
        sampling_strategy=0.5,
        random_state=42,
    )
    balanced_features, balanced_labels = undersampler.fit_resample(
        oversampled_features,
        oversampled_labels,
    )

    # Возвращаем категориальные значения обратно в исходный вид для дерева.
    restored_features = pd.DataFrame(balanced_features, columns=feature_names)
    if save_categorical:
        if categorical_feature_names and encoder is not None:
            restored_features[categorical_feature_names] = encoder.inverse_transform(
                restored_features[categorical_feature_names]
            )

    return np.array(restored_features, dtype=object), np.array(balanced_labels)


def convert_categorical_features(
    x: np.ndarray, encoder, feature_inds: list[int]
) -> np.ndarray:
    x[:, feature_inds] = encoder.transform(x[:, feature_inds])
    return x
