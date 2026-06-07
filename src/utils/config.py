import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: str = None):
        # Корінь проекту — завжди відносно цього файлу
        self.project_root = Path(__file__).parent.parent.parent

        if config_path is None:
            config_path = self.project_root / "configs" / "config.yaml"
        self.config_path = Path(config_path)
        self._config = self._load()

    def _load(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфіг не знайдено: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def resolve_path(self, *keys, default: str = "") -> Path:
        """
        Повертає абсолютний шлях відносно кореня проекту.
        Приклад: config.resolve_path("data", "raw_path")
        """
        relative = self.get(*keys, default=default)
        return self.project_root / relative

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    @property
    def data(self):
        return self._config.get("data", {})

    @property
    def model(self):
        return self._config.get("model", {})

    @property
    def training(self):
        return self._config.get("training", {})

    @property
    def features(self):
        return self._config.get("features", {})


config = Config()