
"""Train a small classifier and serve holdout predictions over Flask."""

from __future__ import annotations

# подключаем библиотеки для работы с файлами, данными и моделью
import os
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from flask import Flask, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# берём настройки из .env, а если их нет, используем значения по умолчанию
DATA_PATH = Path(os.getenv("DATA_PATH", "/home/DATA/clean_data.csv"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "species").strip()
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# создаём flask-сервер
app = Flask(__name__)

# здесь будет храниться результат, чтобы не обучать модель при каждом запросе
_result: dict[str, Any] | None = None
_result_signature: tuple[int, int] | None = None
_result_lock = Lock()


# преобразуем значения в обычный формат python, если это нужно
def _to_json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


# функция для обучения модели и получения предсказаний
def train_and_predict() -> dict[str, Any]:

    # проверяем, получили ли мы csv от артёма
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Clean data is not available at {DATA_PATH}. Transfer clean_data.csv from notebook A first."
        )
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    # читаем csv и на всякий случай ещё раз очищаем данные
    data = pd.read_csv(DATA_PATH, skip_blank_lines=False)
    data = data.dropna(how="all").dropna().copy()
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].astype("string").str.strip().str.lower()
    data = data.reset_index(drop=True)

    # проверяем, что данные подходят для обучения
    if data.empty:
        raise ValueError("clean_data.csv contains no usable rows.")
    if TARGET_COLUMN not in data.columns:
        available = ", ".join(map(str, data.columns))
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} is missing. Available columns: {available}. "
            "Set TARGET_COLUMN in node_b/.env."
        )

    # отделяем признаки ирисов от их настоящих видов
    features = data.drop(columns=[TARGET_COLUMN])
    labels = data[TARGET_COLUMN]
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if labels.nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two classes.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified split.")

    # делим данные на обучающую и тестовую части
    row_indices = pd.Series(range(len(data)))
    try:
        x_train, x_test, y_train, y_test, _, test_indices = train_test_split(
            features,
            labels,
            row_indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot split the dataset into train and test parts: {exc}") from exc

    # определяем, какие столбцы числовые, а какие содержат текст
    numeric_columns = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x_train.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", "passthrough", numeric_columns))
    if categorical_columns:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ))

    # создаём модель дерева решений
    model = Pipeline(
        steps=[
            ("features", ColumnTransformer(transformers=transformers)),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    # обучаем модель на 120 строках
    model.fit(x_train, y_train)

    # проверяем её на 30 строках, которые не использовали для обучения
    predictions = model.predict(x_test)

    # собираем предсказания вместе с номерами строк
    prediction_rows = [
        {
            "row_index": int(row_index),
            "prediction": str(_to_json_value(prediction)),
        }
        for row_index, prediction in zip(test_indices.tolist(), predictions.tolist())
    ]

    # возвращаем результат, включая точность модели
    return {
        "status": "success",
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "predictions": prediction_rows,
    }


# сохраняем результат и пересчитываем его, только если csv изменился
def get_result() -> dict[str, Any]:
    global _result, _result_signature
    stat = DATA_PATH.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if _result is None or _result_signature != signature:
        with _result_lock:
            stat = DATA_PATH.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if _result is None or _result_signature != signature:
                _result = train_and_predict()
                _result_signature = signature
    return dict(_result)


# адрес /health нужен для проверки сервера и наличия csv
@app.get("/health")
def health():
    return jsonify({"status": "ok", "data_ready": DATA_PATH.is_file()})


# адрес /predict возвращает результаты модели в формате json
@app.get("/predict")
def predict():
    try:
        return jsonify(get_result())
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422


# запускаем сервер на порту 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
