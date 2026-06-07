import sys
from pathlib import Path
from loguru import logger
from src.utils.config import config


def setup_logger() -> logger:
    # Прибираємо стандартний handler
    logger.remove()

    # Рівень логування з конфігу
    level = config.get("logging", "level", default="INFO")

    # Логи в консоль
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )

    # Логи у файл
    log_path = config.resolve_path("logging", "log_path", default="reports/logs/")
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "ddos_{time:YYYY-MM-DD}.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="1 day",      # Новий файл щодня
        retention="30 days",   # Зберігати 30 днів
        encoding="utf-8"
    )

    return logger


# Глобальний екземпляр — імпортуй звідси
log = setup_logger()