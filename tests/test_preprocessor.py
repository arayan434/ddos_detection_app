from src.data.loader import load_dataset, drop_unnecessary_columns
from src.data.preprocessor import Preprocessor

# Завантажуємо тестову вибірку
df = load_dataset(sample_rate=0.01, files=['DrDoS_NTP.csv', 'UDPLag.csv'])
df = drop_unnecessary_columns(df)

# Запускаємо передобробку
preprocessor = Preprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process(df)

print(f"\nX_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Ознак:   {len(preprocessor.feature_columns)}")
print(f"Маппінг: {preprocessor.label_mapping}")

print(preprocessor.feature_columns)