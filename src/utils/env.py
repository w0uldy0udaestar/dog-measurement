"""환경 분기 유틸리티. clother-v2의 ENV_NAME 패턴 채용."""

import os
from pathlib import Path

from dotenv import load_dotenv


def setup_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    env_name = os.environ.get("ENV_NAME", "laptop")
    return env_name


def get_device():
    import torch

    device_str = os.environ.get("DEVICE", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_config(config_name: str) -> dict:
    import yaml

    config_path = get_project_root() / "configs" / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)
