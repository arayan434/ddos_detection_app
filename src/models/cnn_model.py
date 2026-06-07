import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(
            self,
            input_size: int,
            dropout: float = 0.3
    ):
        super().__init__("CNN")

        self.conv_layers = nn.Sequential(
            # Перший conv блок
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Другий conv блок
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # Розраховуємо розмір після conv шарів
        conv_output_size = (input_size // 4) * 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)
        # Додаємо channel dimension: (batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.classifier(x).squeeze(1)