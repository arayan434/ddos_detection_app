import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3
    ):
        super().__init__("LSTM")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Якщо вхід має розмірність (batch, input_size), додаємо вимір seq_len
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Беремо останній часовий крок
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(1)