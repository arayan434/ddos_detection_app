import sys
import yaml
import subprocess
import os
import time
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QMessageBox, QLineEdit, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer  # Додано QTimer
from PyQt6.QtGui import QTextCursor


class TrainingProcess(QThread):
    # ... (Клас TrainingProcess залишається абсолютно без змін) ...
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.process = None
        self._is_running = True

    def run(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
            for line in self.process.stdout:
                if not self._is_running:
                    self.process.terminate()
                    break
                self.log_signal.emit(line.strip())
            self.process.wait()
            self.finished_signal.emit(self.process.returncode)
        except Exception as e:
            self.log_signal.emit(f"ПОМИЛКА ЗАПУСКУ: {e}")
            self.finished_signal.emit(-1)

    def stop(self):
        self._is_running = False
        if self.process:
            self.process.terminate()


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config_path = Path("configs/config.yaml")
        self.script_path = Path("tests/test_training.py")
        self.worker = None
        self.start_time = None

        # Ініціалізація таймера для оновлення інтерфейсу
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)

        self.init_ui()
        self.load_current_config()

    def init_ui(self):
        layout = QVBoxLayout(self)

        settings_group = QGroupBox("Параметри навчання")
        form_layout = QFormLayout()

        self.txt_data_files = QLineEdit()
        self.txt_data_files.setPlaceholderText("Оберіть один або декілька CSV файлів...")
        self.btn_browse = QPushButton("Обрати файли...")
        self.btn_browse.clicked.connect(self.browse_data_files)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.txt_data_files)
        path_layout.addWidget(self.btn_browse)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(16, 1024)
        self.spin_batch.setSingleStep(16)

        self.spin_sample = QDoubleSpinBox()
        self.spin_sample.setRange(0.1, 100.0)
        self.spin_sample.setSingleStep(1.0)
        self.spin_sample.setSuffix(" %")

        form_layout.addRow("Файли датасету (CSV):", path_layout)
        form_layout.addRow("Кількість епох:", self.spin_epochs)
        form_layout.addRow("Розмір батчу:", self.spin_batch)
        form_layout.addRow("Відсоток даних:", self.spin_sample)
        settings_group.setLayout(form_layout)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Розпочати навчання")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self.start_training)

        self.btn_stop = QPushButton("⏹ Скасувати")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)

        # === НОВЕ: Мітка часу ===
        self.lbl_timer = QLabel("Витрачено часу: 00:00")
        self.lbl_timer.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px;")

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()  # Відштовхує таймер праворуч
        btn_layout.addWidget(self.lbl_timer)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")

        layout.addWidget(settings_group)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Логи процесу (stdout):"))
        layout.addWidget(self.log_console)

    def update_timer_display(self):
        """Оновлює текстову мітку кожну секунду."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed), 60)
            self.lbl_timer.setText(f"Витрачено часу: {minutes:02d}:{seconds:02d}")

    def browse_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Оберіть файли", "", "CSV Files (*.csv);;All Files (*)"
        )
        if files:
            normalized_paths = [str(Path(f).resolve()) for f in files]
            self.txt_data_files.setText("; ".join(normalized_paths))

    def load_current_config(self):
        if not self.config_path.exists(): return
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        try:
            self.txt_data_files.clear()
            self.spin_epochs.setValue(config.get('training', {}).get('num_epochs', 50))
            self.spin_batch.setValue(config.get('training', {}).get('batch_size', 64))
            self.spin_sample.setValue(config.get('data', {}).get('sample_rate', 1.0))
        except Exception:
            pass

    def save_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if 'training' not in config: config['training'] = {}
        if 'data' not in config: config['data'] = {}

        paths = [p.strip() for p in self.txt_data_files.text().split(";") if p.strip()]
        config['data']['raw_files'] = paths
        config['training']['num_epochs'] = self.spin_epochs.value()
        config['training']['batch_size'] = self.spin_batch.value()
        config['data']['sample_rate'] = self.spin_sample.value()

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    def start_training(self):
        if not self.script_path.exists() or not self.txt_data_files.text().strip():
            QMessageBox.warning(self, "Увага", "Оберіть файли та перевірте скрипт.")
            return

        self.save_config()
        self.log_console.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.spin_epochs.setEnabled(False)
        self.spin_batch.setEnabled(False)
        self.spin_sample.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.txt_data_files.setEnabled(False)

        # Запускаємо таймер
        self.start_time = time.time()
        self.timer.start(1000)  # Оновлення кожні 1000 мс (1 сек)

        self.worker = TrainingProcess(str(self.script_path))
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.timer.stop()  # Зупиняємо таймер
            self.append_log("\n[!] Процес навчання перервано користувачем.")

    def append_log(self, text):
        self.log_console.append(text)
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    def on_training_finished(self, return_code):
        self.timer.stop()  # Зупиняємо таймер

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.spin_epochs.setEnabled(True)
        self.spin_batch.setEnabled(True)
        self.spin_sample.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.txt_data_files.setEnabled(True)

        # Фіксуємо фінальний час
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        if return_code == 0:
            self.append_log(f"\n=== НАВЧАННЯ УСПІШНО ЗАВЕРШЕНО (Витрачено: {time_str}) ===")
            self.prompt_restart(time_str)
        else:
            self.append_log(f"\n=== ПОМИЛКА (Код: {return_code}) ===")

    def prompt_restart(self, time_str):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Успіх")
        msg_box.setText(
            f"Модель успішно оновлено!\n"
            f"Витрачений час: {time_str}\n\n"
            "Щоб система почала використовувати нові ваги, програму необхідно перезавантажити."
        )
        msg_box.setIcon(QMessageBox.Icon.Information)
        btn_restart = msg_box.addButton("Перезавантажити зараз", QMessageBox.ButtonRole.AcceptRole)
        msg_box.addButton("Зроблю це пізніше", QMessageBox.ButtonRole.RejectRole)

        msg_box.exec()
        if msg_box.clickedButton() == btn_restart:
            self.restart_application()

    def restart_application(self):
        self.append_log("\nПерезавантаження програми...")
        subprocess.Popen([sys.executable] + sys.argv)
        QApplication.quit()