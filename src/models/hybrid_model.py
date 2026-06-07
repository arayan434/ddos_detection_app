import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class HybridModel(BaseModel):
    """
    Гібридна CNN + LSTM модель.
    CNN витягує локальні патерни,
    LSTM аналізує часові залежності.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3
    ):
        super().__init__("CNN+LSTM Hybrid")

        # CNN блок — витягує ознаки
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM блок — аналізує послідовності
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Класифікатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)

        # CNN: (batch, 1, input_size) → (batch, 128, input_size)
        x = x.unsqueeze(1)
        x = self.cnn(x)

        # Перестановка для LSTM: (batch, input_size, 128)
        x = x.permute(0, 2, 1)

        # LSTM: (batch, input_size, 128) → (batch, input_size, hidden_size)
        x, _ = self.lstm(x)

        # Беремо останній крок
        x = x[:, -1, :]

        return self.classifier(x).squeeze(1)