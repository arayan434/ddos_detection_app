import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from src.utils.logger import log


class BaseModel(nn.Module, ABC):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def count_parameters(self) -> int:
        """Підраховує кількість параметрів моделі"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Зберігає ваги моделі"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        log.success(f"Модель збережено: {save_path}")

    def load(self, path: str, device: torch.device) -> None:
        """Завантажує ваги моделі"""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Файл моделі не знайдено: {load_path}")
        self.load_state_dict(torch.load(load_path, map_location=device))
        log.success(f"Модель завантажено: {load_path}")

    def summary(self) -> None:
        """Виводить інформацію про модель"""
        log.info(f"Модель: {self.model_name}")
        log.info(f"Параметрів: {self.count_parameters():,}")
        log.info(f"Архітектура:\n{self}")