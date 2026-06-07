import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.logger import log
from src.utils.config import config


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.label_mapping = None

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Прибирає пробіли з назв колонок"""
        df.columns = df.columns.str.strip()
        log.info("Назви колонок очищено від пробілів")
        return df

    def handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Замінює inf, -inf та NaN на коректні значення"""
        # Замінюємо нескінченності на NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Рахуємо кількість NaN
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            log.warning(f"Знайдено {nan_count:,} NaN значень — замінюємо медіаною")
            df = df.fillna(df.median(numeric_only=True))

        log.info("Некоректні значення оброблено")
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодує мітки:
        BENIGN → 0
        Всі атаки → 1 (бінарна класифікація)
        """
        df['Label_binary'] = df['Label'].apply(
            lambda x: 0 if x == 'BENIGN' else 1
        )

        # Зберігаємо маппінг для інформації
        unique_labels = df['Label'].unique()
        self.label_mapping = {
            label: (0 if label == 'BENIGN' else 1)
            for label in unique_labels
        }

        log.info(f"Маппінг міток: {self.label_mapping}")
        log.info(f"Розподіл: {df['Label_binary'].value_counts().to_dict()}")
        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вибирає числові ознаки для навчання"""
        # Виключаємо нечислові та службові колонки
        exclude_cols = ['Label', 'Label_binary']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [
            col for col in numeric_cols
            if col not in exclude_cols
        ]

        log.info(f"Вибрано {len(self.feature_columns)} ознак для навчання")
        return df

    def remove_correlated_features(
            self,
            df: pd.DataFrame,
            threshold: float = 0.95
    ) -> pd.DataFrame:
        """Видаляє сильно корельовані ознаки"""
        feature_df = df[self.feature_columns]
        corr_matrix = feature_df.corr().abs()

        # Знаходимо пари з кореляцією вище порогу
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > threshold)
        ]

        if to_drop:
            self.feature_columns = [
                col for col in self.feature_columns
                if col not in to_drop
            ]
            log.info(f"Видалено {len(to_drop)} корельованих ознак")
            log.info(f"Залишилось {len(self.feature_columns)} ознак")

        return df

    def scale_features(
            self,
            X_train: np.ndarray,
            X_val: np.ndarray,
            X_test: np.ndarray
    ) -> tuple:
        """Нормалізує ознаки за допомогою StandardScaler"""
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        log.info("Ознаки нормалізовано (StandardScaler)")
        return X_train, X_val, X_test

    def balance_classes(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ) -> tuple:
        """Балансує класи за допомогою SMOTE"""
        before = pd.Series(y_train).value_counts().to_dict()
        log.info(f"До балансування: {before}")

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        after = pd.Series(y_resampled).value_counts().to_dict()
        log.info(f"Після балансування: {after}")

        return X_resampled, y_resampled

    def split_data(
            self,
            df: pd.DataFrame
    ) -> tuple:
        """Розбиває дані на train/val/test"""
        test_size = config.get("data", "test_size", default=0.2)
        val_size = config.get("data", "val_size", default=0.1)

        X = df[self.feature_columns].values
        y = df['Label_binary'].values

        # Спочатку відділяємо test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # Потім val від решти
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=42,
            stratify=y_temp
        )

        log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(
            self,
            X_train, X_val, X_test,
            y_train, y_val, y_test
    ) -> None:
        """Зберігає розбиті дані"""
        splits_path = config.resolve_path("data", "splits_path", default="data/splits/")
        splits_path.mkdir(parents=True, exist_ok=True)

        np.save(splits_path / "X_train.npy", X_train)
        np.save(splits_path / "X_val.npy", X_val)
        np.save(splits_path / "X_test.npy", X_test)
        np.save(splits_path / "y_train.npy", y_train)
        np.save(splits_path / "y_val.npy", y_val)
        np.save(splits_path / "y_test.npy", y_test)

        log.success(f"Дані збережено у {splits_path}")

    def process(self, df: pd.DataFrame, balance: bool = True) -> tuple:
        log.info("=== Початок передобробки ===")

        # Перевірка на порожній DataFrame
        if df.empty:
            raise ValueError("DataFrame порожній — перевір чи файли знаходяться у data/raw/")

        df = self.clean_column_names(df)
        df = self.handle_invalid_values(df)
        df = self.encode_labels(df)
        df = self.select_features(df)
        df = self.remove_correlated_features(df)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        if balance:
            X_train, y_train = self.balance_classes(X_train, y_train)

        self.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

        log.success("=== Передобробка завершена ===")
        return X_train, X_val, X_test, y_train, y_val, y_test