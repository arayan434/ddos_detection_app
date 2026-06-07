import sys
from pathlib import Path

# Додаємо кореневу директорію проєкту до системного шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualizer import Visualizer
from src.data.loader import load_dataset, drop_unnecessary_columns
from src.data.preprocessor import Preprocessor
from src.models.hybrid_model import HybridModel
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.config import config

# 1. Завантаження та підготовка даних
df = load_dataset()
df = drop_unnecessary_columns(df)

preprocessor = Preprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process(df)

# 2. Ініціалізація моделі
n_features = X_train.shape[1]
model = HybridModel(input_size=n_features)

# 3. Читання параметрів з config.yaml
cfg_epochs = config.get("training", "num_epochs", default=50)
cfg_batch = config.get("training", "batch_size", default=64)

# 4. Навчання (один виклик!)
trainer = Trainer(model, model_type="hybrid")
history = trainer.train(X_train, y_train, X_val, y_val, epochs=cfg_epochs, batch_size=cfg_batch)

# 5. Оцінка ефективності
evaluator = Evaluator(model, trainer.device, model_type="hybrid")
metrics = evaluator.evaluate(X_test, y_test)

print(f"\nФінальні метрики після оптимізації:")
print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
print(f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")

# 6. Автоматична генерація графіків
try:
    print("\nГенерація графіків навчання...")
    viz = Visualizer()
    viz.plot_training_history(history)
    viz.plot_epoch_metrics(history)

    if "confusion_matrix" in metrics:
        viz.plot_confusion_matrix(metrics["confusion_matrix"])

    print("Всі графіки успішно збережено у папку reports/ !")
except Exception as e:
    print(f"Помилка при генерації графіків: {e}")