import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from src.utils.logger import log


class Evaluator:
    def __init__(self, model, device: torch.device, model_type: str = "hybrid"):
        self.model = model
        self.device = device
        self.model_type = model_type

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Отримує передбачення моделі"""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                outputs = self.model(X_batch)
                all_preds.extend(outputs.cpu().numpy())

        return np.array(all_preds)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Повна оцінка моделі"""
        log.info("=== Оцінка моделі ===")

        probs = self.predict(X_test)
        preds = (probs >= 0.5).astype(int)

        # Метрики
        metrics = {
            "accuracy":  float(np.mean(preds == y_test)),
            "precision": float(precision_score(y_test, preds)),
            "recall":    float(recall_score(y_test, preds)),
            "f1":        float(f1_score(y_test, preds)),
            "roc_auc":   float(roc_auc_score(y_test, probs)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }

        log.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        log.info(f"Precision: {metrics['precision']:.4f}")
        log.info(f"Recall:    {metrics['recall']:.4f}")
        log.info(f"F1 Score:  {metrics['f1']:.4f}")
        log.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        log.info(f"\n{classification_report(y_test, preds, target_names=['BENIGN', 'DDoS'])}")

        return metrics