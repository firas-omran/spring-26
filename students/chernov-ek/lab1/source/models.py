import numpy as np

from evaluators import (
    calculate_gini_impurity,
    calculate_information_gain,
    calculate_weighted_gini_impurity,
)


class Node:
    """
    Хранит данные одного узла дерева решений.

    Attributes:
        value (str | None): Значение узла или метка ребра. По умолчанию: None.
        next (Node | None): Следующий узел для ребра. По умолчанию: None.
        children (list[Node] | None): Дочерние рёбра узла. По умолчанию: None.
        feature_id (int | None): Индекс признака для разбиения. По умолчанию: None.
        threshold (float | None): Порог для числового признака. По умолчанию: None.
        is_numeric (bool): Флаг числового признака. По умолчанию: False.
        prediction (int | None): Прогноз большинства классов в узле. По умолчанию: None.
        class_probabilities (numpy.ndarray | None): Вероятности классов в узле. По умолчанию: None.
        branch_probabilities (dict[str, float] | None): Вероятности перехода по ветвям. По умолчанию: None.
        missing_values (set[str] | None): Маркеры пропущенных значений признака. По умолчанию: None.

    Fallbacks:
        Узел может быть листом, если children равно None.
    """

    def __init__(self) -> None:
        """
        Инициализирует пустой узел дерева решений.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Атрибуты узла заполняются начальными значениями.

        Fallbacks:
            Все связи и значения остаются None до построения дерева.
        """
        # Значение хранит имя признака в узле или класс в листе.
        self.value: str | None = None

        # Следующий узел используется для представления ребра дерева.
        self.next: Node | None = None

        # Дочерние элементы содержат рёбра от текущего узла.
        self.children: list[Node] | None = None

        # Параметры разбиения заполняются для внутренних узлов.
        self.feature_id: int | None = None
        self.threshold: float | None = None
        self.is_numeric = False

        # Прогноз большинства нужен для неизвестных ветвей при предсказании.
        self.prediction: int | None = None

        # Вероятности классов нужны для вероятностного обхода дерева.
        self.class_probabilities: np.ndarray | None = None

        # Вероятности ветвей нужны для распределения пропусков.
        self.branch_probabilities: dict[str, float] | None = None

        # Маркеры пропусков нужны для отличия неизвестного значения от категории.
        self.missing_values: set[str] | None = None


class DecisionTreeClassifier:
    """
    Реализует классификатор дерева решений с ID3-подобным алгоритмом.

    Attributes:
        features (numpy.ndarray): Матрица признаков обучающей выборки. По умолчанию: нет.
        feature_names (list[str]): Названия признаков. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов обучающей выборки. По умолчанию: нет.
        node (Node | None): Корневой узел дерева. По умолчанию: None.
        gini (float): Неоднородность Джини исходной выборки. По умолчанию: вычисляется.
        classes_ (numpy.ndarray): Упорядоченный список классов. По умолчанию: вычисляется.
        missing_value_map (dict[str, set[str]]): Маркеры пропусков по признакам. По умолчанию: фиксированные значения для датасета.

    Fallbacks:
        При неизвестной ветке используется прогноз большинства классов текущего узла.
    """

    def __init__(
        self,
        features: np.ndarray,
        feature_names: list[str],
        labels: np.ndarray,
        pass_processing_type: str = "none",
    ) -> None:
        """
        Сохраняет обучающую выборку и начальные параметры дерева.

        Parameters:
            features (numpy.ndarray): Матрица признаков. По умолчанию: нет.
            feature_names (list[str]): Названия признаков. По умолчанию: нет.
            labels (numpy.ndarray): Метки классов. По умолчанию: нет.
            pass_processing_type (str): Тип обработки пропусков. По умолчанию: "none".

        Returns:
            None: Экземпляр классификатора получает начальное состояние.

        Fallbacks:
            Корневой узел остаётся None до вызова id3().
        """
        # Сохраняем обучающие данные в виде массивов NumPy.
        self.features = np.array(features)
        self.feature_names = list(feature_names)
        self.labels = np.array(labels)

        # Корень дерева создаётся во время обучения.
        self.node: Node | None = None
        self.gini = calculate_gini_impurity(self.labels)
        self.classes_ = np.unique(self.labels)
        self.pass_processing_type = pass_processing_type
        self.missing_value_map = {
            "gender": {"Other"},
            "smoking_history": {"No Info"},
        }

    def _get_missing_values(self, feature_id: int) -> set[str]:
        """
        Возвращает набор маркеров пропуска для признака.

        Parameters:
            feature_id (int): Индекс признака. По умолчанию: нет.

        Returns:
            set[str]: Набор строковых маркеров пропуска.

        Fallbacks:
            Для признаков без специальных маркеров возвращается пустое множество.
        """
        # В базовом режиме пропуски считаются отдельной категорией.
        if self.pass_processing_type != "weight":
            return set()

        # Ищем маркеры пропуска по имени признака.
        feature_name = str(self.feature_names[feature_id])
        return set(self.missing_value_map.get(feature_name, set()))

    def _majority_class(
        self,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        """
        Возвращает самый частый класс в наборе меток с учётом весов.

        Parameters:
            labels (numpy.ndarray): Метки классов. По умолчанию: нет.
            weights (numpy.ndarray): Веса объектов. По умолчанию: нет.

        Returns:
            int: Метка класса с максимальным суммарным весом.

        Fallbacks:
            При равенстве весов выбирается первый класс после сортировки self.classes_.
        """
        # Суммируем веса по классам и выбираем наиболее вероятный класс.
        class_weights = np.array(
            [np.sum(weights[labels == label]) for label in self.classes_],
            dtype=float,
        )
        return int(self.classes_[np.argmax(class_weights)])

    def _calculate_class_probabilities(
        self,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Вычисляет вероятности классов в узле по взвешенным объектам.

        Parameters:
            labels (numpy.ndarray): Метки классов. По умолчанию: нет.
            weights (numpy.ndarray): Веса объектов. По умолчанию: нет.

        Returns:
            numpy.ndarray: Вероятности классов в порядке self.classes_.

        Fallbacks:
            При нулевой сумме весов возвращается нулевой вектор.
        """
        # Суммируем веса по классам в фиксированном порядке.
        class_weights = np.array(
            [np.sum(weights[labels == label]) for label in self.classes_],
            dtype=float,
        )
        total_weight = np.sum(class_weights)
        if total_weight == 0:
            return np.zeros(len(self.classes_), dtype=float)

        return class_weights / total_weight

    def _get_feature_max_information_gain(
        self,
        row_ids: list[int],
        row_weights: np.ndarray,
        feature_ids: list[int],
    ) -> tuple[str, int, float | None, bool] | None:
        """
        Находит признак с максимальным приростом информации.

        Parameters:
            row_ids (list[int]): Индексы объектов текущего узла. По умолчанию: нет.
            row_weights (numpy.ndarray): Веса объектов текущего узла. По умолчанию: нет.
            feature_ids (list[int]): Индексы доступных признаков. По умолчанию: нет.

        Returns:
            tuple[str, int, float | None, bool] | None: Описание лучшего признака.

        Fallbacks:
            Если список признаков пуст, возвращается None.
        """
        # Инициализируем лучший признак отсутствующим значением.
        best_feature = None
        best_gain = -1

        # Проверяем каждый доступный признак на текущем подмножестве строк.
        for feature_id in feature_ids:
            gain, threshold, is_numeric = calculate_information_gain(
                self.features[row_ids],
                self.labels[row_ids],
                feature_id,
                row_weights,
                self._get_missing_values(feature_id),
            )

            # Запоминаем признак, который даёт максимальное улучшение.
            if gain > best_gain:
                best_gain = gain
                best_feature = (
                    str(self.feature_names[feature_id]),
                    feature_id,
                    threshold,
                    is_numeric,
                )

        return best_feature

    def id3(self) -> None:
        """
        Строит дерево решений по алгоритму ID3 с критерием Джини.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Корневой узел сохраняется в self.node.

        Fallbacks:
            Если признаки не дают разбиения, дерево завершается листом большинства.
        """
        # Начинаем построение со всех строк и всех признаков.
        row_ids = list(range(len(self.features)))
        row_weights = np.ones(len(self.features), dtype=float)
        feature_ids = list(range(len(self.feature_names)))
        self.node = self._id3_recursive(
            row_ids,
            row_weights,
            feature_ids,
            self.node,
        )

    def _id3_recursive(
        self,
        row_ids: list[int],
        row_weights: np.ndarray,
        feature_ids: list[int],
        node: Node | None,
    ) -> Node:
        """
        Рекурсивно строит поддерево для выбранных объектов и признаков.

        Parameters:
            row_ids (list[int]): Индексы объектов в текущем узле. По умолчанию: нет.
            row_weights (numpy.ndarray): Веса объектов в текущем узле. По умолчанию: нет.
            feature_ids (list[int]): Индексы доступных признаков. По умолчанию: нет.
            node (Node | None): Узел для заполнения. По умолчанию: None.

        Returns:
            Node: Заполненный узел дерева.

        Fallbacks:
            При чистом узле, отсутствии признаков или порога создаётся лист.
        """
        # Создаём узел, если рекурсивный вызов получил пустую ссылку.
        if node is None:
            node = Node()

        # Запоминаем распределение классов и резервный прогноз для узла.
        labels_in_node = self.labels[row_ids]
        node.class_probabilities = self._calculate_class_probabilities(
            labels_in_node,
            row_weights,
        )
        node.prediction = self._majority_class(labels_in_node, row_weights)

        # Останавливаем построение, если узел уже чистый по взвешенной метрике.
        if calculate_weighted_gini_impurity(labels_in_node, row_weights) == 0:
            node.value = str(node.prediction)
            return node

        # Останавливаем построение, если признаки закончились.
        if len(feature_ids) == 0:
            node.value = str(node.prediction)
            return node

        # Выбираем лучший признак для разбиения текущего узла.
        best_feature = self._get_feature_max_information_gain(
            row_ids,
            row_weights,
            feature_ids,
        )
        if best_feature is None:
            node.value = str(node.prediction)
            return node

        # Записываем параметры выбранного признака во внутренний узел.
        best_feature_name, best_feature_id, threshold, is_numeric = best_feature
        node.value = best_feature_name
        node.feature_id = best_feature_id
        node.threshold = threshold
        node.is_numeric = is_numeric
        node.children = []
        node.branch_probabilities = None
        node.missing_values = self._get_missing_values(best_feature_id)

        # Исключаем использованный признак для дочерних разбиений.
        next_feature_ids = [
            feature_id for feature_id in feature_ids if feature_id != best_feature_id
        ]

        # Формируем бинарные ветки для числового признака.
        if is_numeric:
            if threshold is None:
                node.value = str(node.prediction)
                node.children = None
                return node

            numeric_values = self.features[row_ids, best_feature_id].astype(float)
            splits = [
                (
                    f"<= {threshold:.4f}",
                    [
                        (row_id, weight)
                        for row_id, value, weight in zip(
                            row_ids,
                            numeric_values,
                            row_weights,
                        )
                        if value <= threshold
                    ],
                ),
                (
                    f"> {threshold:.4f}",
                    [
                        (row_id, weight)
                        for row_id, value, weight in zip(
                            row_ids,
                            numeric_values,
                            row_weights,
                        )
                        if value > threshold
                    ],
                ),
            ]
        else:
            # Отделяем наблюдаемые категории от пропусков для вероятностного разбиения.
            feature_values = self.features[row_ids, best_feature_id]
            observed_mask = ~np.isin(feature_values, list(node.missing_values))
            observed_values = feature_values[observed_mask]
            observed_row_ids = np.array(row_ids)[observed_mask]
            observed_weights = row_weights[observed_mask]

            # Если для признака нет наблюдаемых значений, узел становится листом.
            if len(observed_values) == 0:
                node.value = str(node.prediction)
                node.children = None
                return node

            # Оцениваем вероятности ветвей по наблюдаемой части данных.
            node.branch_probabilities = {}
            total_observed_weight = np.sum(observed_weights)
            unique_values = np.unique(observed_values)
            for value in unique_values:
                value_weight = np.sum(observed_weights[observed_values == value])
                node.branch_probabilities[str(value)] = value_weight / total_observed_weight

            # Готовим пропущенные значения для распределения по всем ветвям.
            missing_row_ids = np.array(row_ids)[~observed_mask]
            missing_weights = row_weights[~observed_mask]
            splits = []
            for value in unique_values:
                value_row_ids = observed_row_ids[observed_values == value]
                value_weights = observed_weights[observed_values == value]
                branch_probability = node.branch_probabilities[str(value)]

                # Копируем пропуски в ветвь с весом, равным вероятности этой ветви.
                child_row_ids = np.concatenate([value_row_ids, missing_row_ids])
                child_weights = np.concatenate(
                    [value_weights, missing_weights * branch_probability]
                )
                child_rows = list(zip(child_row_ids.tolist(), child_weights.tolist()))
                splits.append((str(value), child_rows))

        # Рекурсивно строим дочерние узлы для каждого разбиения.
        for value, child_rows in splits:
            child = Node()
            child.value = str(value)
            node.children.append(child)

            # Преобразуем список пар обратно в индексы и веса для дочернего узла.
            child_row_ids = [row_id for row_id, _ in child_rows]
            child_row_weights = np.array(
                [weight for _, weight in child_rows],
                dtype=float,
            )

            # Пустая ветка получает лист с прогнозом большинства текущего узла.
            if len(child_row_ids) == 0 or np.sum(child_row_weights) == 0:
                child.next = Node()
                child.next.value = str(node.prediction)
                child.next.prediction = node.prediction
                child.next.class_probabilities = node.class_probabilities
            else:
                child.next = self._id3_recursive(
                    child_row_ids,
                    child_row_weights,
                    next_feature_ids.copy(),
                    child.next,
                )

        return node

    def _predict_proba_one(self, row: np.ndarray, node: Node) -> np.ndarray:
        """
        Предсказывает вероятности классов для одного объекта.

        Parameters:
            row (numpy.ndarray): Значения признаков одного объекта. По умолчанию: нет.
            node (Node): Корневой или текущий узел дерева. По умолчанию: нет.

        Returns:
            numpy.ndarray: Вероятности классов в порядке self.classes_.

        Fallbacks:
            Если обход невозможен, возвращается распределение текущего узла.
        """
        # Лист или некорректный узел возвращают локальное распределение классов.
        if not node.children or node.feature_id is None:
            if node.class_probabilities is None:
                return np.zeros(len(self.classes_), dtype=float)

            return node.class_probabilities

        # Для числового признака выбираем ровно одну подходящую ветвь.
        if node.is_numeric:
            if node.threshold is None:
                return node.class_probabilities

            value = float(row[node.feature_id])
            branch_value = (
                node.children[0].value
                if value <= node.threshold
                else node.children[1].value
            )

            for child in node.children:
                if child.value == branch_value and child.next is not None:
                    return self._predict_proba_one(row, child.next)

            return node.class_probabilities

        # Для пропуска усредняем прогнозы по вероятностям ветвей.
        feature_value = str(row[node.feature_id])
        if node.missing_values and feature_value in node.missing_values:
            if not node.branch_probabilities:
                return node.class_probabilities

            probabilities = np.zeros(len(self.classes_), dtype=float)
            for child in node.children:
                if child.next is None:
                    continue

                branch_probability = node.branch_probabilities.get(child.value, 0)
                probabilities += (
                    branch_probability * self._predict_proba_one(row, child.next)
                )

            if np.sum(probabilities) == 0:
                return node.class_probabilities

            return probabilities / np.sum(probabilities)

        # Для обычной категории выбираем соответствующую ветвь.
        for child in node.children:
            if child.value == feature_value and child.next is not None:
                return self._predict_proba_one(row, child.next)

        return node.class_probabilities

    def _predict_one(self, row: np.ndarray, node: Node) -> int | None:
        """
        Предсказывает класс для одного объекта.

        Parameters:
            row (numpy.ndarray): Значения признаков одного объекта. По умолчанию: нет.
            node (Node): Корневой или текущий узел дерева. По умолчанию: нет.

        Returns:
            int | None: Предсказанная метка класса.

        Fallbacks:
            Если вероятности не удалось вычислить, возвращается прогноз узла.
        """
        # Преобразуем вероятности классов в итоговую метку.
        probabilities = self._predict_proba_one(row, node)
        if np.sum(probabilities) == 0:
            return node.prediction

        return int(self.classes_[np.argmax(probabilities)])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Предсказывает классы для одного или нескольких объектов.

        Parameters:
            features (numpy.ndarray): Матрица признаков или один объект. По умолчанию: нет.

        Returns:
            numpy.ndarray: Массив предсказанных меток классов.

        Fallbacks:
            Если дерево не обучено, выбрасывается ValueError.
        """
        # Проверяем, что дерево построено перед предсказанием.
        if self.node is None:
            raise ValueError("The tree has not been trained. Call id3() first.")

        # Приводим один объект к матрице из одной строки.
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Предсказываем класс независимо для каждой строки.
        return np.array([self._predict_one(row, self.node) for row in features])

    def print_tree(self) -> None:
        """
        Печатает дерево решений в читаемом иерархическом виде.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Структура дерева выводится в консоль.

        Fallbacks:
            Если дерево не построено, печатается понятное сообщение.
        """
        # Проверяем, что дерево построено перед выводом структуры.
        if not self.node:
            print("Дерево не построено. Сначала вызовите id3().")
            return

        def format_node(node: Node) -> str:
            """
            Формирует подпись узла для вывода дерева.

            Parameters:
                node (Node): Узел дерева решений. По умолчанию: нет.

            Returns:
                str: Читаемая подпись узла.

            Fallbacks:
                Если значение узла пустое, используется строка "пустой узел".
            """
            # Лист отображаем как итоговый класс, внутренний узел как признак.
            if not node.children:
                return f"класс: {node.value if node.value is not None else node.prediction}"

            return f"признак: {node.value if node.value is not None else 'пустой узел'}"

        def print_node(
            node: Node,
            line_prefix: str = "",
            children_prefix: str = "",
            branch: str = "",
        ) -> None:
            """
            Рекурсивно печатает узел и его дочерние ветви.

            Parameters:
                node (Node): Текущий узел дерева. По умолчанию: нет.
                line_prefix (str): Отступ строки текущего узла. По умолчанию: "".
                children_prefix (str): Отступ дочерних строк. По умолчанию: "".
                branch (str): Подпись ребра от родителя. По умолчанию: "".

            Returns:
                None: Узел и его потомки печатаются в консоль.

            Fallbacks:
                Если дочерняя ссылка отсутствует, ветка помечается как пустая.
            """
            # Печатаем текущую строку дерева с подписью условия перехода.
            label = format_node(node)
            print(f"{line_prefix}{branch}{label}")

            # Продолжаем рекурсию для дочерних ветвей текущего узла.
            if node.children:
                for index, child in enumerate(node.children):
                    is_last = index == len(node.children) - 1
                    connector = "`-- " if is_last else "|-- "
                    next_children_prefix = children_prefix + (
                        "    " if is_last else "|   "
                    )
                    condition = f"если {node.value} {child.value}: "

                    # Выводим пустую ветку явно, чтобы структура не выглядела оборванной.
                    if child.next is None:
                        print(f"{children_prefix}{connector}{condition}пустая ветка")
                    else:
                        print_node(
                            child.next,
                            children_prefix,
                            next_children_prefix,
                            connector + condition,
                        )

        # Начинаем вывод с заголовка и корневого узла дерева.
        print("Дерево решений")
        print_node(self.node)
