import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import log
from src.utils.config import config
from src.data.preprocessor import Preprocessor


class DDoSDetector:
    def __init__(self, model, model_type: str = "hybrid"):
        self.model = model
        self.model_type = model_type
        self.preprocessor = Preprocessor()

        # Визначаємо пристрій
        device_cfg = config.get("training", "device", default="auto")
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        self.model = self.model.to(self.device)
        self.model.eval()

        log.info(f"Детектор ініціалізовано | Пристрій: {self.device}")

    def _preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """Підготовка вхідних даних для inference"""
        # Очищаємо назви колонок
        df = self.preprocessor.clean_column_names(df)
        df = self.preprocessor.handle_invalid_values(df)

        # Вибираємо тільки потрібні ознаки
        if self.preprocessor.feature_columns is None:
            raise ValueError("Препроцесор не навчений! Спочатку запустіть process()")

        available = [col for col in self.preprocessor.feature_columns if col in df.columns]
        if not available:
            raise ValueError("Жодна з необхідних ознак не знайдена у вхідних даних")

        X = df[available].values
        X = self.preprocessor.scaler.transform(X)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Повертає ймовірності для кожного зразка"""
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor)

        return probs.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Повертає бінарні передбачення (0=BENIGN, 1=DDoS)"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def detect_from_dataframe(
            self,
            df: pd.DataFrame,
            threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Основний метод виявлення атак з DataFrame.
        Повертає DataFrame з результатами.
        """
        log.info(f"Аналіз {len(df):,} мережевих потоків...")

        X = self._preprocess_input(df)
        probs = self.predict_proba(X)
        predictions = (probs >= threshold).astype(int)

        # Формуємо результат
        result = pd.DataFrame({
            "probability": probs,
            "prediction": predictions,
            "label": ["DDoS" if p == 1 else "BENIGN" for p in predictions],
            "risk_level": [self._get_risk_level(p) for p in probs]
        })

        # Статистика
        total = len(result)
        ddos_count = result["prediction"].sum()
        benign_count = total - ddos_count

        log.info(f"Результати аналізу:")
        log.info(f"  Всього потоків: {total:,}")
        log.info(f"  BENIGN:         {benign_count:,} ({benign_count/total*100:.1f}%)")
        log.info(f"  DDoS:           {ddos_count:,} ({ddos_count/total*100:.1f}%)")

        if ddos_count > 0:
            log.warning(f"⚠️ Виявлено {ddos_count:,} підозрілих потоків!")
        else:
            log.success("✅ Атак не виявлено")

        return result

    def _get_risk_level(self, probability: float) -> str:
        """Визначає рівень ризику за ймовірністю"""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_summary(self, results: pd.DataFrame) -> dict:
        """Повертає зведену статистику результатів"""
        total = len(results)
        ddos = results["prediction"].sum()

        return {
            "total_flows":    total,
            "benign_count":   int(total - ddos),
            "ddos_count":     int(ddos),
            "ddos_percent":   round(ddos / total * 100, 2),
            "avg_probability": round(float(results["probability"].mean()), 4),
            "risk_distribution": results["risk_level"].value_counts().to_dict(),
            "is_under_attack": bool(ddos > 0)
        }

    @classmethod
    def load_from_checkpoint(
            cls,
            model_class,
            checkpoint_path: str,
            input_size: int,
            model_type: str = "hybrid"
    ) -> "DDoSDetector":
        """Завантажує модель з файлу та створює детектор"""
        model = model_class(input_size=input_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint не знайдено: {checkpoint_path}")

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log.success(f"Модель завантажено з {checkpoint_path}")

        return cls(model, model_type=model_type)

    def detect_realtime(self, features: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Легковигий метод для інференсу одного мережевого потоку в реальному часі.

        Args:
            features: Вектор ознак розмірністю (n_features, ...) або (1, n_features)
            threshold: Поріг класифікації

        Returns:
            Словник із результатами класифікації потоку
        """
        # 1. Перевірка та зміна форми масиву для StandardScaler (очікує 2D)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 2. Нормалізація (критично важливо використовувати той самий scaler, що і при навчанні)
        try:
            scaled_features = self.preprocessor.scaler.transform(features)
        except Exception as e:
            log.error(f"Помилка нормалізації ознак: {e}")
            return {"status": "error", "message": str(e)}

        # 3. Конвертація в PyTorch тензор та перенесення на потрібний пристрій (CPU/GPU)
        x_tensor = torch.FloatTensor(scaled_features).to(self.device)

        # 4. Пряме поширення (Forward pass)
        with torch.no_grad():
            prob = self.model(x_tensor).item()  # .item() повертає чисте значення float

        # 5. Класифікація та визначення ризику
        prediction = 1 if prob >= threshold else 0

        return {
            "status": "success",
            "probability": prob,
            "prediction": "DDoS" if prediction == 1 else "BENIGN",
            "risk_level": self._get_risk_level(prob)
        }