import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import log
from src.utils.config import config

DROP_COLUMNS = [
    'Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port',
    ' Destination IP', ' Destination Port', ' Timestamp',
    'SimillarHTTP', ' Inbound'
]

DATASET_FILES = [
    'DrDoS_DNS.csv', 'DrDoS_LDAP.csv', 'DrDoS_MSSQL.csv', 'DrDoS_NTP.csv',
    'DrDoS_NetBIOS.csv', 'DrDoS_SNMP.csv', 'DrDoS_SSDP.csv', 'DrDoS_UDP.csv',
    'Syn.csv', 'TFTP.csv', 'UDPLag.csv'
]

def load_single_file(
        filepath: Path,
        sample_rate: float = 0.1,
        chunksize: int = 50000
) -> pd.DataFrame:
    """Завантажує один CSV файл по частинах за вказаним абсолютним або відносним шляхом."""
    log.info(f"Завантаження файлу: {filepath.name}")
    chunks = []
    total_rows = 0

    try:
        for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
            sampled = chunk.sample(frac=sample_rate, random_state=42)
            chunks.append(sampled)
            total_rows += len(chunk)

        df = pd.concat(chunks, ignore_index=True)
        log.success(f"{filepath.name}: {total_rows:,} рядків → вибрано {len(df):,}")
        return df
    except Exception as e:
        log.error(f"Помилка при завантаженні {filepath.name}: {e}")
        return pd.DataFrame()


def load_dataset(
        data_path: str = None,
        sample_rate: float = None,
        files: list = None
) -> pd.DataFrame:
    # Визначаємо базову директорію для відносних шляхів
    if data_path is None:
        data_dir = config.resolve_path("data", "raw_path", default="data/raw/")
    else:
        data_dir = Path(data_path)
        if not data_dir.is_absolute():
            data_dir = config.project_root / data_dir

    # === ЖОРСТКА ПРИВ'ЯЗКА ДО GUI ===
    # Ігноруємо старі аргументи скриптів і беремо файли ТІЛЬКИ з конфігурації
    files_to_load = config.get("data", "raw_files", default=[])

    if not files_to_load:
        files_to_load = DATASET_FILES
        log.info(f"Список з GUI порожній. Беремо всі базові файли з: {data_dir}")
    else:
        log.info("Використовується список файлів, обраний у графічному інтерфейсі.")

    # 2. Визначаємо sample_rate з урахуванням формату GUI (відсотки 0.1-100.0)
    if sample_rate is None:
        cfg_sample = config.get("data", "sample_rate", default=10.0)
        sample_rate = cfg_sample / 100.0 if cfg_sample > 1.0 else cfg_sample

    log.info(
        f"Параметри завантаження | Кількість файлів: {len(files_to_load)} | Частка вибірки: {sample_rate * 100:.2f}%")

    all_dataframes = []

    for file_item in files_to_load:
        filepath = Path(file_item)
        if not filepath.is_absolute():
            filepath = data_dir / filepath

        if not filepath.exists():
            log.warning(f"Файл не знайдено за шляхом: {filepath}")
            continue

        df = load_single_file(filepath, sample_rate=sample_rate)
        if not df.empty:
            all_dataframes.append(df)

    if not all_dataframes:
        log.error("Жоден із зазначених мережевих логів не був завантажений!")
        return pd.DataFrame()

    combined = pd.concat(all_dataframes, ignore_index=True)
    log.success(f"Всього завантажено для навчання: {len(combined):,} рядків")
    log.info(f"Розподіл класів у вибірці:\n{combined[' Label'].value_counts()}")

    return combined

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Видалено службових колонок: {len(cols_to_drop)}")
    return df

def save_raw_combined(df: pd.DataFrame, output_path: str = None) -> None:
    if output_path is None:
        output_path = config.get("data", "processed_path", default="data/processed/")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "combined_raw.csv"
    df.to_csv(filepath, index=False)
    log.success(f"Збережено об'єднаний лог: {filepath} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")