import argparse

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

from data_loader import balance_train_dataset, load_archive, load_df, convert_categorical_features
from models import DecisionTreeClassifier as CustomDecisionTreeClassifier


def parse_arguments() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска обучения.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        argparse.Namespace: Пространство имён с параметрами запуска.

    Fallbacks:
        Если параметр не указан, используется режим обработки пропусков "none".
    """
    # Создаём парсер для настройки запуска из терминала.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pass-processing-type",
        default="none",
        choices=["none", "weight"],
        help="Тип обработки пропусков: none или weight.",
    )
    parser.add_argument(
        "--show-tree",
        default="no",
        choices=["no", "yes"],
        help="Печатать дерево в конце: no или yes.",
    )
    parser.add_argument(
        "--use-reduction",
        default="no",
        choices=["no", "yes"],
        help="Включить редукцию дерева: no или yes.",
    )
    parser.add_argument(
        "--use-scklearn-model",
        default="no",
        choices=["no", "yes"],
        help="Использовать модель из Sklearn: no или yes.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Загружает данные, обучает дерево решений и выводит accuracy.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Результат оценки выводится в консоль.

    Fallbacks:
        Ошибки загрузки данных и обучения передаются вызывающему коду.
    """
    # Разбираем параметры запуска перед обучением модели.
    arguments = parse_arguments()

    # Загружаем архив и читаем таблицу с датасетом.
    load_archive()
    dataframe = load_df()

    # Разделяем признаки и целевую переменную по последнему столбцу.
    feature_names = dataframe.columns[:-1]
    label_name = dataframe.columns[-1]

    # Формируем стратифицированные обучающую и тестовую выборки.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataframe[feature_names]),
        np.array(dataframe[label_name]),
        test_size=0.2,
        random_state=42,
        stratify=dataframe[label_name],
    )

    # Выделяем валидационную часть только для редукции, если она запрошена.
    x_reduction = np.empty((0, x_train.shape[1]), dtype=x_train.dtype)
    y_reduction = np.empty(0, dtype=y_train.dtype)
    if arguments.use_reduction == "yes":
        x_train, x_reduction, y_train, y_reduction = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

    # Балансируем только обучающую выборку, не затрагивая тестовую.
    encoder = OrdinalEncoder()
    x_train, y_train = balance_train_dataset(
        x_train,
        y_train,
        feature_names,
        encoder,
        arguments.use_scklearn_model == "no",
    )

    if arguments.use_scklearn_model == "yes":
        x_test = convert_categorical_features(x_test, encoder, [0, 4])
        
        # Создаем модель решающего дерева
        model = DecisionTreeClassifier(
            criterion="gini",
            random_state=42
        )

        # Обучаем модель
        model.fit(x_train, y_train)

        # Получаем предсказания
        y_predicted = model.predict(x_test)
    else:
        # Обучаем собственную реализацию дерева решений.
        tree_classifier = CustomDecisionTreeClassifier(
            features=x_train,
            feature_names=feature_names,
            labels=y_train,
            pass_processing_type=arguments.pass_processing_type,
        )
        tree_classifier.id3()

        # Запускаем редукцию по отложенной валидационной выборке.
        if arguments.use_reduction == "yes":
            tree_classifier.reduce(x_reduction, y_reduction)

        # Оцениваем качество классификации на тестовой выборке.
        y_predicted = tree_classifier.predict(x_test)

        # Печатаем дерево только по явному запросу пользователя.
        if arguments.show_tree == "yes":
            tree_classifier.print_tree()
    
    print(classification_report(y_test, y_predicted, target_names=['Class 0', 'Class 1']))


# Запускаем сценарий только при прямом вызове файла.
if __name__ == "__main__":
    main()
