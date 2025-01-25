import logging
from typing import Tuple

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Pipeline,
    pipeline,
)


def get_device(device_arg: str) -> str:
    """
    Determine the device to use for computation.

    Args:
        device_arg (str): Device specified by the user.

    Returns:
        str: The device to use ('cpu', 'cuda', or 'mps').
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Get the appropriate torch data type based on the device.

    Args:
        device (str): The device being used.

    Returns:
        torch.dtype: The torch data type to use.
    """
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def load_model_and_processor(
    model_name: str, device: str, torch_dtype: torch.dtype
) -> Tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
    """
    Load the pre-trained model and processor.

    Args:
        model_name (str): Name or path of the pre-trained model.
        device (str): Device to load the model onto.
        torch_dtype (torch.dtype): Data type for the model parameters.

    Returns:
        Tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]: The loaded model and processor.
    """
    logging.info(f"Loading model '{model_name}'")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def create_pipeline(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    torch_dtype: torch.dtype,
    device: str,
) -> Pipeline:
    """
    Create a speech recognition pipeline.

    Args:
        model (AutoModelForSpeechSeq2Seq): The pre-trained model.
        processor (AutoProcessor): The processor associated with the model.
        torch_dtype (torch.dtype): Data type for the model parameters.
        device (str): Device for computation.

    Returns:
        Pipeline: The speech recognition pipeline.
    """
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.startswith("cuda") else -1,
        chunk_length_s=30,
        batch_size=16,
    )


def process_audio(
    input_file: str, language: str, task: str, pipeline: Pipeline
) -> dict:
    """
    Process the audio file using the provided pipeline.

    Args:
        input_file (str): Path to the input audio file.
        language (str): Language of the audio.
        task (str): Task to perform ('transcribe' or 'translate').
        pipeline (Pipeline): The speech recognition pipeline.

    Returns:
        dict: The result containing transcriptions and other metadata.
    """
    logging.info(f"Processing file: {input_file}")
    logging.info(f"Starting task: {task}")

    try:
        result = pipeline(
            input_file,
            return_timestamps="word",
            generate_kwargs={"language": language, "task": task},
        )
        logging.info(f"Task '{task}' completed successfully")
        return result
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        raise
