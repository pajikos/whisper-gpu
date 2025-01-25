# core/audio_processor.py
from typing import Dict, Any, Optional
import torch
from transformers import Pipeline
import logging
from config.settings import CONFIG

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def process_audio(
            pipeline: Pipeline,
            audio_path: str,
            language: str,
            task: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio file using the provided pipeline.

        Args:
            pipeline: Initialized transformers Pipeline
            audio_path: Path to the audio file
            language: Language code for processing
            task: Task type (transcribe or translate)
            **kwargs: Additional arguments for the pipeline

        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing audio file: {audio_path}")
            logger.info(f"Task: {task}, Language: {language}")

            with open(audio_path, 'rb') as audio_file:
                result = pipeline(
                    audio_file,
                    return_timestamps=True,
                    generate_kwargs={
                        "language": language,
                        "task": task
                    },
                    **kwargs
                )

            logger.info("Audio processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e
