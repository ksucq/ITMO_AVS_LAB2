
"""Clean a CSV, send it to notebook B, and request predictions."""

from __future__ import annotations

# подключаем библиотеки, которые нужны для работы программы
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_iris

# настраиваем вывод сообщений о работе программы
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# проверяем настройку из файла .env, например USE_IRIS_FALLBACK
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# если исходного csv нет, создаём файл с данными об ирисах
def create_iris_csv(path: Path, target_column: str) -> pd.DataFrame:
    """Create the fallback labeled dataset when no instructor CSV is present."""
    iris = load_iris(as_frame=True)
    frame = iris.data.copy()
    frame[target_column] = iris.target.map(lambda value: iris.target_names[int(value)])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logging.info("Created fallback Iris CSV at %s (%d rows).", path, len(frame))
    return frame


# читаем исходный csv, а если его нет, пробуем создать данные ирисов
def load_raw_data(raw_path: Path, target_column: str) -> pd.DataFrame:
    if raw_path.is_file():
        return pd.read_csv(raw_path, skip_blank_lines=False)
    if _env_bool("USE_IRIS_FALLBACK", True):
        return create_iris_csv(raw_path, target_column)
    raise FileNotFoundError(
        f"Raw CSV not found at {raw_path}. Add the instructor CSV or enable USE_IRIS_FALLBACK."
    )


# очищаем данные: убираем пустые строки и приводим текст к нижнему регистру
def clean_data(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, int, int]:
    original_rows = len(frame)
    cleaned = frame.dropna(how="all").dropna().copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()
    cleaned = cleaned.reset_index(drop=True)

    # проверяем, что после очистки остались данные и нужный столбец
    if cleaned.empty:
        raise ValueError("No rows remain after removing blank or incomplete rows.")
    if target_column not in cleaned.columns:
        available = ", ".join(map(str, cleaned.columns))
        raise ValueError(
            f"Target column {target_column!r} is missing. Available columns: {available}. "
            "Set TARGET_COLUMN in node_a/.env."
        )
    if cleaned[target_column].nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two classes.")

    removed_rows = original_rows - len(cleaned)
    return cleaned, original_rows, removed_rows


# собираем настройки подключения по ssh и проверяем наличие ключа
def _ssh_base(ssh_key: Path, known_hosts: Path, port: int) -> list[str]:
    if not ssh_key.is_file():
        raise FileNotFoundError(f"SSH private key not found at {ssh_key}.")
    if not known_hosts.is_file():
        raise FileNotFoundError(
            f"SSH known_hosts file not found at {known_hosts}. Connect to notebook B from "
            "notebook A once and verify its host fingerprint first."
        )
    return [
        "-i",
        str(ssh_key),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=10",
    ]


# отправляем очищенный csv на ноутбук Б через scp
def transfer_csv(clean_path: Path) -> None:
    host = os.getenv("NODE_B_HOST", "").strip()
    user = os.getenv("NODE_B_USER", "").strip()
    remote_dir = os.getenv("NODE_B_DATA_DIR", "").strip()
    if not host or not user or not remote_dir:
        raise ValueError(
            "Set NODE_B_HOST, NODE_B_USER, and NODE_B_DATA_DIR in node_a/.env."
        )
    if not remote_dir.startswith("/"):
        raise ValueError("NODE_B_DATA_DIR must be an absolute path on notebook B.")

    ssh_key = Path(os.getenv("SSH_KEY_PATH", "/run/ssh/lab2_ed25519"))
    known_hosts = Path(os.getenv("SSH_KNOWN_HOSTS", "/run/ssh/known_hosts"))
    port = int(os.getenv("SSH_PORT", "22"))
    remote = f"{user}@{host}"
    remote_dir_arg = shlex.quote(remote_dir)

    # сначала создаём нужную папку на ноутбуке Б
    ssh_command = ["ssh", *_ssh_base(ssh_key, known_hosts, port), remote,
                   f"mkdir -p -- {remote_dir_arg}"]
    subprocess.run(ssh_command, check=True, text=True, capture_output=True)

    # потом копируем туда clean_data.csv
    scp_command = [
        "scp",
        "-i",
        str(ssh_key),
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=10",
        str(clean_path),
        f"{remote}:{remote_dir.rstrip('/')}/clean_data.csv",
    ]
    subprocess.run(scp_command, check=True, text=True, capture_output=True)
    logging.info("Transferred clean_data.csv to %s:%s.", remote, remote_dir)


# отправляем запрос к flask-серверу на ноутбуке Б
def request_predictions() -> dict[str, Any]:
    host = os.getenv("NODE_B_HOST", "").strip()
    if not host:
        raise ValueError("Set NODE_B_HOST in node_a/.env.")

    url = f"http://{host}:5000/predict"
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-connrefused",
        "--retry-delay",
        "2",
        "--max-time",
        "15",
        url,
    ]
    response = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(response.stdout)
    if payload.get("status") != "success":
        raise RuntimeError(f"Notebook B returned an error: {payload}")
    return payload


# основная часть программы: читаем, очищаем, передаём csv и получаем ответ
def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "/home/DATA"))
    raw_path = Path(os.getenv("RAW_DATA_PATH", str(data_dir / "raw_data.csv")))
    clean_path = Path(os.getenv("CLEAN_DATA_PATH", str(data_dir / "clean_data.csv")))
    target_column = os.getenv("TARGET_COLUMN", "species").strip()

    raw = load_raw_data(raw_path, target_column)
    cleaned, original_rows, removed_rows = clean_data(raw, target_column)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    logging.info(
        "Cleaned %d rows; removed %d blank or incomplete rows. Saved %s.",
        len(cleaned),
        removed_rows,
        clean_path,
    )
    logging.info("Feature columns: %s; target: %s.",
                 ", ".join(column for column in cleaned.columns if column != target_column),
                 target_column)

    transfer_csv(clean_path)
    payload = request_predictions()
    payload["source_rows"] = original_rows
    payload["removed_rows"] = removed_rows
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# отсюда начинается выполнение программы
if __name__ == "__main__":
    main()
