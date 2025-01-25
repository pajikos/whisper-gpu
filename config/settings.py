# config/settings.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingConfig:
    chunk_length_s: int = 30
    batch_size: int = 16
    default_language: str = "en"
    default_model: str = "openai/whisper-large-v3"

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    mono_channels: int = 1
    mp3_bitrate: str = "320k"

CONFIG = {
    "processing": ProcessingConfig(),
    "audio": AudioConfig(),
}
