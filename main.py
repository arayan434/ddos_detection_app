import sys
import joblib
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from src.models.hybrid_model import HybridModel
from src.detection.detector import DDoSDetector
from src.core.engine import TrafficAnalyzerEngine
from src.gui.main_window import MainWindow
from src.utils.logger import log


def load_ai_components():
    """Завантажує ваги моделі та препроцесор."""
    log.info("Ініціалізація AI компонентів...")

    # 1. Завантажуємо препроцесор
    prep_path = Path("saved_models/preprocessor.pkl")
    if not prep_path.exists():
        raise FileNotFoundError(f"Не знайдено файл препроцесора: {prep_path}")
    preprocessor = joblib.load(prep_path)

    # 2. Ініціалізуємо модель (забезпечуємо ту ж кількість ознак, що й при навчанні - 48)
    n_features = 48
    model = HybridModel(input_size=n_features)

    # 3. Ініціалізуємо детектор
    detector = DDoSDetector(model, model_type="hybrid")
    detector.preprocessor = preprocessor

    # Завантажуємо ваги (best_hybrid.pth)
    weights_path = Path("saved_models/best_hybrid.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Не знайдено ваги моделі: {weights_path}")

    detector.model.load_state_dict(
        __import__("torch").load(weights_path, map_location=detector.device, weights_only=True)
    )
    detector.model.eval()

    log.success("AI компоненти успішно завантажено.")
    return detector


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Сучасний системний стиль

    try:
        # Ініціалізуємо важкі компоненти до запуску GUI
        detector = load_ai_components()

        # Фабрика для передачі в GUI
        def engine_factory(interface):
            return TrafficAnalyzerEngine(interface, detector, update_interval=1.0)

        window = MainWindow(engine_factory)
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        log.error(f"Критична помилка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
