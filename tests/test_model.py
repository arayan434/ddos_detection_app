import torch
from src.models.lstm_model import LSTMModel
from src.models.cnn_model import CNNModel
from src.models.hybrid_model import HybridModel

INPUT_SIZE = 48  # кількість ознак
BATCH_SIZE = 64

# Тестовий тензор
x = torch.randn(BATCH_SIZE, INPUT_SIZE)

# LSTM очікує 3D вхід (batch, seq_len, features)
x_lstm = x.unsqueeze(1)

# Тест LSTM
lstm = LSTMModel(input_size=INPUT_SIZE)
out = lstm(x_lstm)
lstm.summary()
print(f"LSTM вихід: {out.shape}")

# Тест CNN
cnn = CNNModel(input_size=INPUT_SIZE)
out = cnn(x)
cnn.summary()
print(f"CNN вихід: {out.shape}")

# Тест Hybrid
hybrid = HybridModel(input_size=INPUT_SIZE)
out = hybrid(x)
hybrid.summary()
print(f"Hybrid вихід: {out.shape}")