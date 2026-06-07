from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableView, QHeaderView, QMessageBox,
    QComboBox, QSlider, QTabWidget
)
from PyQt6.QtCore import Qt, QAbstractTableModel, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QFileDialog
from scapy.config import conf

from src.gui.training_tab import TrainingTab

class SignalEmitter(QObject):
    result_ready = pyqtSignal(dict)


class TrafficTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["Flow ID", "Протокол", "Пакети", "Байти", "Прогноз", "Ймовірність", "Ризик"]

    def data(self, index, role):
        if not index.isValid(): return None
        row_data = self._data[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            keys = ["flow_id", "protocol", "packets", "bytes", "prediction", "probability", "risk_level"]
            val = row_data[keys[col]]
            if keys[col] == "probability": return f"{val:.2%}"
            return str(val)

        elif role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor(Qt.GlobalColor.black))


        elif role == Qt.ItemDataRole.BackgroundRole:
            prediction = row_data.get("prediction", "")
            risk = str(row_data.get("risk_level", "")).upper()
            prob = float(row_data.get("probability", 0.0))
            # 1. Червоний (Критичний): якщо мережа класифікувала потік як DDoS,
            if prediction == "DDoS" or any(w in risk for w in ["КРИТИЧ", "CRITICAL", "ВИСОК", "HIGH"]):
                return QBrush(QColor(255, 200, 200))
                # 2. Жовтий (Підозрілий): якщо ймовірність атаки вище 50% (але менше порогу),
            elif prob >= 0.5 or any(w in risk for w in ["СЕРЕДН", "MEDIUM", "ПІДОЗР", "SUSPICIOUS"]):
                return QBrush(QColor(255, 255, 200))
                # 3. Зелений (Нормальний трафік)
            return QBrush(QColor(200, 255, 200))

        return None

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._headers)

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def add_row(self, row_data: dict):
        for i, existing in enumerate(self._data):
            if existing["flow_id"] == row_data["flow_id"]:
                self._data[i] = row_data
                self.dataChanged.emit(self.index(i, 0), self.index(i, len(self._headers) - 1))
                return
        self.beginInsertRows(self.index(len(self._data), 0), len(self._data), len(self._data))
        self._data.append(row_data)
        self.endInsertRows()


class MonitorTab(QWidget):
    def __init__(self, engine_factory):
        super().__init__()
        self.engine_factory = engine_factory
        self.engine = None
        self.emitter = SignalEmitter()
        self.emitter.result_ready.connect(self.update_table)
        self.detected_ddos_flows = set()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        control_layout = QHBoxLayout()
        self.lbl_interface = QLabel("Мережевий інтерфейс:")
        self.combo_interface = QComboBox()
        interfaces = [iface.name for iface in conf.ifaces.values()]
        self.combo_interface.addItems(sorted(list(set(interfaces))))

        self.btn_start = QPushButton("▶ Запустити")
        self.btn_start.clicked.connect(self.start_monitoring)
        self.btn_stop = QPushButton("⏹ Зупинити")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_monitoring)

        self.btn_export = QPushButton("💾 Експорт звіту (CSV)")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        self.lbl_threshold = QLabel("Поріг тривоги: 90%")
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(50, 99)
        self.slider_threshold.setValue(90)
        self.slider_threshold.valueChanged.connect(self.update_threshold)

        control_layout.addWidget(self.lbl_interface)
        control_layout.addWidget(self.combo_interface)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_export)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.lbl_threshold)
        control_layout.addWidget(self.slider_threshold)

        self.table_view = QTableView()
        self.table_model = TrafficTableModel()
        self.table_view.setModel(self.table_model)
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        # === НОВЕ: Нижня панель статусів ===
        status_layout = QHBoxLayout()

        self.lbl_status = QLabel("Статус: Очікування...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")

        self.lbl_threat_count = QLabel("Виявлено загроз: 0")
        self.lbl_threat_count.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")

        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()  # Розштовхує елементи по краях
        status_layout.addWidget(self.lbl_threat_count)

        layout.addLayout(control_layout)
        layout.addWidget(self.table_view)
        layout.addLayout(status_layout)  # Додаємо весь status_layout замість одного lbl_status

    # ... (методи update_threshold та export_report залишаються без змін) ...
    def update_threshold(self, value):
        self.lbl_threshold.setText(f"Поріг тривоги: {value}%")
        if self.engine: self.engine.threshold = value / 100.0

    def export_report(self):
        if not self.table_model._data:
            return QMessageBox.warning(self, "Увага", "Таблиця порожня, немає даних для експорту.")
        file_path, _ = QFileDialog.getSaveFileName(self, "Зберегти звіт", "traffic_report.csv", "CSV Files (*.csv)")
        if file_path:
            import csv
            try:
                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.table_model._headers)
                    for row in self.table_model._data:
                        writer.writerow(
                            [row["flow_id"], row["protocol"], row["packets"], row["bytes"], row["prediction"],
                             f'{row["probability"]:.2%}', row["risk_level"]])
                QMessageBox.information(self, "Успіх", f"Звіт успішно збережено:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти файл:\n{e}")

    def start_monitoring(self):
        interface = self.combo_interface.currentText()
        if not interface: return QMessageBox.warning(self, "Помилка", "Не знайдено інтерфейсів!")
        try:
            # === ОНОВЛЕНО: Очищуємо множину при новому запуску ===
            self.detected_ddos_flows.clear()
            self.lbl_threat_count.setText("Виявлено загроз: 0")

            self.engine = self.engine_factory(interface)
            self.engine.threshold = self.slider_threshold.value() / 100.0
            self.engine.on_result_ready = lambda data: self.emitter.result_ready.emit(data)
            self.engine.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("Статус: Активний моніторинг...")
            self.lbl_status.setStyleSheet("font-weight: bold; color: green;")
        except Exception as e:
            QMessageBox.critical(self, "Помилка ініціалізації", str(e))

    def stop_monitoring(self):
        if self.engine: self.engine.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Статус: Моніторинг зупинено.")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")

    def update_table(self, data: dict):
        self.table_model.add_row(data)
        self.table_view.scrollToBottom()

        # === ВИПРАВЛЕНО: Рахуємо лише унікальні Flow ID ===
        if data["prediction"] == "DDoS":
            if data["flow_id"] not in self.detected_ddos_flows:
                self.detected_ddos_flows.add(data["flow_id"])
                self.lbl_threat_count.setText(f"Виявлено загроз: {len(self.detected_ddos_flows)}")


class MainWindow(QMainWindow):
    """Головне вікно з підтримкою вкладок."""

    def __init__(self, engine_factory):
        super().__init__()
        self.setWindowTitle("DDoS Detection System - Control Center")
        self.resize(1100, 700)

        # Створюємо віджет вкладок
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Ініціалізуємо вкладки
        self.monitor_tab = MonitorTab(engine_factory)
        self.training_tab = TrainingTab()

        # Додаємо вкладки у вікно
        self.tabs.addTab(self.monitor_tab, "Монітор Трафіку")
        self.tabs.addTab(self.training_tab, "Навчання Моделі")