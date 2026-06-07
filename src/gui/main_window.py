from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableView, QHeaderView, QMessageBox,
    QComboBox, QSlider, QTabWidget, QFrame
)
from PyQt6.QtCore import Qt, QAbstractTableModel, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QFileDialog

# НОВЕ: Прямий імпорт функції збору інтерфейсів Windows
from scapy.arch.windows import get_windows_if_list

from src.gui.training_tab import TrainingTab


class SignalEmitter(QObject):
    result_ready = pyqtSignal(dict)


class TrafficTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["Flow ID", "Протокол", "Пакети", "Байти", "Прогноз", "Ймовірність", "Ризик"]
        self.current_threshold = 0.90  # Ініціалізація відстеження порогу

    def data(self, index, role):
        if not index.isValid(): return None
        row_data = self._data[index.row()]
        col = index.column()

        prob = float(row_data.get("probability", 0.0))
        is_ddos = prob >= self.current_threshold

        if role == Qt.ItemDataRole.DisplayRole:
            keys = ["flow_id", "protocol", "packets", "bytes", "prediction", "probability", "risk_level"]
            key = keys[col]

            # Текст генерується на льоту замість використання старих значень
            if key == "probability":
                return f"{prob:.2%}"
            elif key == "prediction":
                return "DDoS" if is_ddos else "Норма"
            elif key == "risk_level":
                if is_ddos:
                    return "Критичний"
                elif prob >= 0.5:
                    return "Середній"
                else:
                    return "Низький"

            return str(row_data.get(key, ""))

        elif role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor(Qt.GlobalColor.black))

        elif role == Qt.ItemDataRole.BackgroundRole:
            if is_ddos:
                return QBrush(QColor(255, 200, 200))
            elif prob >= 0.5:
                return QBrush(QColor(255, 255, 200))
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
        self.combo_interface.setMaximumWidth(400)

        # ОНОВЛЕНО: Наповнення списку з використанням GUID
        interfaces = get_windows_if_list()
        for iface in interfaces:
            if iface.get('ips') or "TP-LINK" in iface.get('description', ''):
                display_name = f"{iface.get('name', 'Unknown')} - {iface.get('description', '')}"
                guid = iface.get('guid')

                # Додаємо префікс, який вимагає драйвер Npcap у Windows
                npf_path = rf"\Device\NPF_{guid}"

                self.combo_interface.addItem(display_name, npf_path)

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
        self.slider_threshold.setMinimumWidth(150)
        self.slider_threshold.valueChanged.connect(self.update_threshold)

        control_layout.addWidget(self.lbl_interface)
        control_layout.addWidget(self.combo_interface)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_export)

        # Відновлена лінія-розділювач
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        control_layout.addSpacing(15)
        control_layout.addWidget(separator)
        control_layout.addSpacing(15)

        control_layout.addWidget(self.lbl_threshold)
        control_layout.addWidget(self.slider_threshold)

        self.table_view = QTableView()
        self.table_model = TrafficTableModel()
        self.table_view.setModel(self.table_model)
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        status_layout = QHBoxLayout()

        self.lbl_status = QLabel("Статус: Очікування...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")

        self.lbl_threat_count = QLabel("Виявлено загроз: 0")
        self.lbl_threat_count.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")

        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        status_layout.addWidget(self.lbl_threat_count)

        layout.addLayout(control_layout)
        layout.addWidget(self.table_view)
        layout.addLayout(status_layout)


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
        # ОНОВЛЕНО: Витягуємо прихований ідентифікатор замість тексту
        interface_guid = self.combo_interface.currentData()

        if not interface_guid:
            return QMessageBox.warning(self, "Помилка", "Не знайдено інтерфейсів!")

        try:
            self.detected_ddos_flows.clear()
            self.lbl_threat_count.setText("Виявлено загроз: 0")

            # Передаємо GUID безпосередньо у двигун перехоплення
            self.engine = self.engine_factory(interface_guid)
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

    def update_threshold(self, value):
        new_threshold = value / 100.0
        self.lbl_threshold.setText(f"Поріг тривоги: {value}%")

        if self.engine:
            self.engine.threshold = new_threshold

        # Передаємо нове значення у таблицю та примусово перемальовуємо її
        self.table_model.current_threshold = new_threshold
        self.table_model.layoutChanged.emit()

        # Тотальний перерахунок історії унікальних загроз
        self.detected_ddos_flows.clear()
        for row in self.table_model._data:
            if float(row.get("probability", 0.0)) >= new_threshold:
                self.detected_ddos_flows.add(row["flow_id"])

        self.lbl_threat_count.setText(f"Виявлено загроз: {len(self.detected_ddos_flows)}")

    def update_table(self, data: dict):
        self.table_model.add_row(data)
        self.table_view.scrollToBottom()

        # Фіксація нової загрози відбувається за актуальним порогом екрана
        prob = float(data.get("probability", 0.0))
        if prob >= self.table_model.current_threshold:
            if data["flow_id"] not in self.detected_ddos_flows:
                self.detected_ddos_flows.add(data["flow_id"])
                self.lbl_threat_count.setText(f"Виявлено загроз: {len(self.detected_ddos_flows)}")


class MainWindow(QMainWindow):
    def __init__(self, engine_factory):
        super().__init__()
        self.setWindowTitle("DDoS Detection System - Control Center")
        self.resize(1100, 700)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.monitor_tab = MonitorTab(engine_factory)
        self.training_tab = TrainingTab()

        self.tabs.addTab(self.monitor_tab, "Монітор Трафіку")
        self.tabs.addTab(self.training_tab, "Навчання Моделі")