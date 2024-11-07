import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg

def get_torch_dtype(device):
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32

def load_model_and_processor(model_name, device, torch_dtype):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def create_pipeline(model, processor, torch_dtype, device):
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        batch_size=16,
    )

def process_audio(input_file, args, pipe):
    print(f"Processing file: {input_file}")
    print(f"Starting {args.task}...")

    with open(input_file, 'rb') as f:
        audio_data = f.read()

    result = pipe(
        audio_data,
        return_timestamps=True,
        generate_kwargs={"language": args.language, "task": args.task}
    )

    print(f"\n{args.task.capitalize()} completed.")
    return result
