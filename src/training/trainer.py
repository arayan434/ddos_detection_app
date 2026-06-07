import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import log
from src.utils.config import config


class Trainer:
    def __init__(self, model, model_type: str = "hybrid"):
        self.model = model
        self.model_type = model_type

        # Визначаємо пристрій
        device_cfg = config.get("training", "device", default="auto")
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        log.info(f"Пристрій: {self.device}")
        if self.device.type == "cuda":
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = self.model.to(self.device)

        # Гіперпараметри з конфігу (training)
        self.lr = config.get("model", "learning_rate", default=0.001)
        self.batch_size = config.get("training", "batch_size", default=64)
        self.epochs = config.get("training", "num_epochs", default=50)
        self.patience = config.get("training", "early_stopping_patience", default=5)
        self.checkpoint_path = config.get("training", "checkpoint_path", default="saved_models/")

        # Оптимізатор та функція втрат
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        # РОЗШИРЕНА Історія навчання
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
            "train_f1": [], "val_f1": [],
            "train_roc_auc": [], "val_roc_auc": []
        }

    def _prepare_dataloader(
            self,
            X: np.ndarray,
            y: np.ndarray,
            shuffle: bool = True
    ) -> DataLoader:
        """Перетворює numpy масиви в DataLoader"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> dict:
        """Запускає одну епоху та збирає розширені метрики"""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for X_batch, y_batch in dataloader:
                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                # Збираємо дані для розрахунку метрик
                probs = outputs.cpu().detach().numpy()
                preds = (probs >= 0.5).astype(float)
                targets = y_batch.cpu().detach().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = total_loss / len(dataloader)

        # Розраховуємо scikit-learn метрики для всієї епохи
        metrics = {
            "loss": avg_loss,
            "acc": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0)
        }

        try:
            metrics["roc_auc"] = roc_auc_score(all_targets, all_probs)
        except ValueError:
            metrics["roc_auc"] = 0.5  # Запобіжник, якщо в батчі опинився лише один клас

        return metrics

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int = None,
            batch_size: int = None
    ) -> dict:
        """Повний цикл навчання з early stopping"""
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size

        log.info("=== Початок навчання ===")
        log.info(f"Модель: {self.model.model_name}")
        log.info(f"Епох: {self.epochs} | Batch: {self.batch_size} | LR: {self.lr}")

        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, y_val, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            # Отримуємо словники з метриками
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            # Scheduler
            self.scheduler.step(val_metrics["loss"])

            # Зберігаємо історію
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_acc"].append(val_metrics["acc"])

            self.history["train_precision"].append(train_metrics["precision"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["train_recall"].append(train_metrics["recall"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["train_roc_auc"].append(train_metrics["roc_auc"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])

            log.info(
                f"Епоха {epoch:3d}/{self.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} / {val_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['acc']:.4f} / {val_metrics['acc']:.4f} | "
                f"F1: {train_metrics['f1']:.4f} / {val_metrics['f1']:.4f}"
            )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                self._save_best_model()
                log.success(f"Нова найкраща модель збережена (епоха {epoch})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.warning(f"Early stopping на епосі {epoch}")
                    break

        log.success("=== Навчання завершено ===")
        self._save_history()
        return self.history

    def _save_best_model(self) -> None:
        """Зберігає найкращу модель"""
        save_dir = config.resolve_path("training", "checkpoint_path", default="saved_models/")
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_{self.model_type}.pth"
        torch.save(self.model.state_dict(), path)

    def _save_history(self) -> None:
        """Зберігає історію навчання у CSV файл для побудови графіків."""
        try:
            history_df = pd.DataFrame(self.history)
            history_df.insert(0, 'epoch', range(1, len(history_df) + 1))

            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = log_dir / f"training_history_{self.model_type}_{timestamp}.csv"

            history_df.to_csv(file_path, index=False)
            log.success(f"Історію навчання збережено: {file_path}")
        except Exception as e:
            log.error(f"Помилка збереження історії: {e}")