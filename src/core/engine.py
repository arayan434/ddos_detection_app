import time
import threading
import csv
from datetime import datetime
from pathlib import Path
from typing import Callable
from src.core.sniffer import LiveTrafficSniffer
from src.detection.detector import DDoSDetector
from src.utils.logger import log


class TrafficAnalyzerEngine:
    def __init__(self, interface: str, detector: DDoSDetector, update_interval: float = 1.0):
        self.interface = interface
        self.detector = detector
        self.update_interval = update_interval
        self.sniffer = LiveTrafficSniffer(interface=self.interface)
        self._running = False
        self.on_result_ready: Callable[[dict], None] = None
        self.threshold = 0.90

    def start(self):
        if self._running: return
        self._running = True
        self._sniff_thread = threading.Thread(target=self.sniffer.start, daemon=True)
        self._sniff_thread.start()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        log.info("Обчислювальне ядро успішно запущено.")

    def stop(self):
        self._running = False
        log.info("Зупинка обчислювального ядра...")

    def _log_incident(self, data: dict):
        """Зберігає інформацію про виявлену атаку у CSV файл."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = log_dir / f"incidents_{date_str}.csv"
        file_exists = file_path.exists()

        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Час", "Flow ID", "Протокол", "Пакети", "Байти", "Прогноз", "Ймовірність"])

                writer.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    data["flow_id"], data["protocol"], data["packets"],
                    data["bytes"], data["prediction"], f'{data["probability"]:.4f}'
                ])
        except Exception as e:
            log.error(f"Помилка запису логу: {e}")

    def _inference_loop(self):
        while self._running:
            time.sleep(self.update_interval)

            # Потокобезпечне отримання списку активних ідентифікаторів
            active_flow_ids = self.sniffer.get_all_active_flow_ids()

            for flow_id in active_flow_ids:
                # Потокобезпечне читання статистики та ознак
                stats, features = self.sniffer.get_flow_stats_and_features(flow_id)

                if not stats or not features is not None:
                    continue

                if stats["packets"] < 5:
                    continue

                result = self.detector.detect_realtime(features, threshold=self.threshold)

                if result.get("status") == "success":
                    gui_data = {
                        "flow_id": flow_id,
                        "protocol": stats["protocol"],
                        "packets": stats["packets"],
                        "bytes": stats["bytes"],
                        "prediction": result["prediction"],
                        "probability": result["probability"],
                        "risk_level": result["risk_level"]
                    }

                    if result["prediction"] == "DDoS":
                        self._log_incident(gui_data)

                    if self.on_result_ready:
                        self.on_result_ready(gui_data)