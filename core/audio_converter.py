# core/audio_converter.py
from pathlib import Path
from typing import Union
import subprocess
import logging
from config.settings import CONFIG

logger = logging.getLogger(__name__)

class AudioConverter:
    @staticmethod
    def ensure_compatible_audio(input_path: Union[str, Path]) -> Path:
        """Ensure audio file compatibility with proper error handling."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() in {'.wav', '.mp3'}:
            return input_path

        if input_path.suffix.lower() in {'.m4a', '.mov'}:
            return AudioConverter.convert_to_mp3(input_path)

        raise ValueError(f"Unsupported audio format: {input_path.suffix}")

    @staticmethod
    def convert_to_mp3(input_path: Path) -> Path:
        """Convert m4a or mov to mp3 with proper error handling and logging."""
        output_path = input_path.with_suffix('.mp3')

        config = CONFIG['audio']

        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ac", str(config.mono_channels),
            "-ar", str(config.sample_rate),
            "-codec:a", "libmp3lame",
            "-b:a", config.mp3_bitrate,
            "-hide_banner",
            "-loglevel", "error",
            str(output_path)
        ]

        try:
            logger.info(f"Converting {input_path} to MP3")
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg conversion failed: {error_msg}")
            raise RuntimeError(f"Audio conversion failed: {error_msg}") from e