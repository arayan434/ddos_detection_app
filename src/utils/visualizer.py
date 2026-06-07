import numpy as np
from src.utils.config import config
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.logger import log


class Visualizer:
    def __init__(self, save_path: str = "reports/"):
        self.save_path = config.resolve_path("logging", "log_path", default="reports/") \
            .parent if save_path == "reports/" else Path(save_path)
        self.save_path = config.project_root / "reports"
        self.save_path.mkdir(parents=True, exist_ok=True)
        plt.style.use("dark_background")
        self.colors = {
            "primary":  "#00D4FF",
            "success":  "#00FF88",
            "danger":   "#FF4444",
            "warning":  "#FFB300",
            "neutral":  "#888888"
        }


    def plot_confusion_matrix(
            self,
            cm: list,
            save: bool = True
    ) -> plt.Figure:
        """Теплова карта confusion matrix"""
        fig, ax = plt.subplots(figsize=(7, 6))

        cm_array = np.array(cm)
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["BENIGN", "DDoS"],
            yticklabels=["BENIGN", "DDoS"],
            ax=ax,
            linewidths=0.5
        )

        ax.set_title("Confusion Matrix", fontsize=14, color="white", pad=15)
        ax.set_ylabel("Справжній клас", color="white")
        ax.set_xlabel("Передбачений клас", color="white")
        plt.tight_layout()

        if save:
            path = self.save_path / "confusion_matrix.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.success(f"Confusion matrix збережено: {path}")

        return fig

    def plot_training_history(
            self,
            history: dict,
            save: bool = True
    ) -> plt.Figure:
        """Базова динаміка навчання: функція втрат та загальна точність."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Динаміка збіжності нейромережевої моделі", fontsize=16, color="white")

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"],
                 color=self.colors["primary"], label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"],
                 color=self.colors["danger"], label="Val Loss",
                 linewidth=2, linestyle="--")
        ax1.set_title("Функція втрат", color="white")
        ax1.set_xlabel("Епоха")
        ax1.set_ylabel("Значення")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, history["train_acc"],
                 color=self.colors["success"], label="Train Acc", linewidth=2)
        ax2.plot(epochs, history["val_acc"],
                 color=self.colors["warning"], label="Val Acc",
                 linewidth=2, linestyle="--")

        all_acc_values = history["train_acc"] + history["val_acc"]
        min_acc = min(all_acc_values) if all_acc_values else 0.9

        if min_acc > 0.95:
            ax2.set_ylim(min_acc - 0.005, 1.002)
        else:
            ax2.set_ylim(max(0.0, min_acc - 0.05), 1.01)

        ax2.set_title("Загальна точність", color="white")
        ax2.set_xlabel("Епоха")
        ax2.set_ylabel("Точність")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.save_path / "training_history.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.success(f"Графік базової динаміки збережено: {path}")

        return fig

    def plot_epoch_metrics(
            self,
            history: dict,
            save: bool = True
    ) -> plt.Figure:
        """Комплексні валідаційні метрики з агресивним фокусуванням."""
        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = range(1, len(history["val_loss"]) + 1)

        metrics_to_plot = {
            "Precision": history.get("val_precision", []),
            "Recall": history.get("val_recall", []),
            "F1 Score": history.get("val_f1", []),
            "ROC-AUC": history.get("val_roc_auc", [])
        }

        line_styles = ["-", "--", "-.", ":"]
        colors = [self.colors["success"], self.colors["primary"], self.colors["warning"], self.colors["danger"]]

        all_plotted_values = []

        for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
            if values:
                color = colors[i % len(colors)]
                style = line_styles[i % len(line_styles)]
                ax.plot(epochs, values, color=color, linestyle=style, label=metric_name, linewidth=2)
                all_plotted_values.extend(values)

        if all_plotted_values:
            min_val = min(all_plotted_values)
            if min_val > 0.98:
                ax.set_ylim(min_val - 0.002, 1.001)
            elif min_val > 0.90:
                ax.set_ylim(min_val - 0.01, 1.005)
            else:
                ax.set_ylim(max(0.0, min_val - 0.05), 1.01)

        ax.set_title("Еволюція комплексних метрик на валідаційній вибірці", fontsize=14, color="white")
        ax.set_xlabel("Епоха", color="white")
        ax.set_ylabel("Значення", color="white")

        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            path = self.save_path / "epoch_metrics.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.success(f"Графік розширених метрик збережено: {path}")

        return fig

    def plot_risk_distribution(
            self,
            results,
            save: bool = True
    ) -> plt.Figure:
        """Кругова діаграма розподілу рівнів ризику"""
        risk_counts = results["risk_level"].value_counts()
        risk_colors = {
            "LOW":      self.colors["success"],
            "MEDIUM":   self.colors["warning"],
            "HIGH":     "#FF8C00",
            "CRITICAL": self.colors["danger"]
        }

        colors = [risk_colors.get(r, self.colors["neutral"]) for r in risk_counts.index]

        fig, ax = plt.subplots(figsize=(7, 6))
        wedges, texts, autotexts = ax.pie(
            risk_counts.values,
            labels=risk_counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )

        for text in texts + autotexts:
            text.set_color("white")

        ax.set_title("Розподіл рівнів ризику", fontsize=14, color="white")
        plt.tight_layout()

        if save:
            path = self.save_path / "risk_distribution.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.success(f"Розподіл ризиків збережено: {path}")

        return fig