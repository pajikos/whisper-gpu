import argparse
from datetime import datetime
from audio_converter import ensure_compatible_audio
from audio_processor import get_device, get_torch_dtype, load_model_and_processor, create_pipeline, process_audio
from utils import format_time_delta, get_output_filenames, write_output_files

def main():
    parser = argparse.ArgumentParser(description="Audio processing with Whisper model")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use for processing")
    parser.add_argument("--language", default="en", help="Language of the audio")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="Model name or path")
    parser.add_argument("--input", required=True, help="Input audio file path (absolute or relative)")
    parser.add_argument("--task", choices=["transcribe", "translate"], required=True, help="Task to perform")
    parser.add_argument("--output", help="Output file path for the result (optional)")

    args = parser.parse_args()

    try:
        start_time = datetime.now()
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        device = get_device(args.device)
        print(f"Using device: {device}")

        torch_dtype = get_torch_dtype(device)
        model, processor = load_model_and_processor(args.model, device, torch_dtype)
        pipe = create_pipeline(model, processor, torch_dtype, device)

        input_file = args.input
        print(f"Ensuring audio file is compatible...")
        input_file = ensure_compatible_audio(input_file)

        result = process_audio(input_file, args, pipe)

        vtt_output, txt_output = get_output_filenames(args)
        write_output_files(result, vtt_output, txt_output)

        end_time = datetime.now()
        processing_time = end_time - start_time

        print(f"VTT result written to {vtt_output}")
        print(f"Plain text result written to {txt_output}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Processing time: {format_time_delta(processing_time)}")
        print("\nResult:")
        print(result)
        print("\nProcessing completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
