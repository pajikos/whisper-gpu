# Audio Transcription with Whisper

This tool provides an easy way to transcribe or translate audio files using OpenAI's Whisper model. It supports various audio formats and provides optimized performance for different hardware configurations, including Mac M-series processors.

## Prerequisites

- Python 3.12
- ffmpeg (required for audio conversion)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic command structure:
```bash
python main.py --input <audio-file> --language <language-code> --task <task-type>
```

### Required Parameters

- `--input`: Path to the input audio file
- `--task`: Type of task to perform (`transcribe` or `translate`)

### Optional Parameters

- `--device`: Computing device to use (`cpu`, `cuda`, `mps`, or `auto` [default])
- `--language`: Language code of the audio (default: "en")
- `--model`: Model name or path (default: "openai/whisper-large-v3")
- `--output`: Custom output file path

### Examples

1. Basic transcription of an English audio file:
```bash
python main.py --input data/audio.mp3 --language en --task transcribe
```

2. Translation to English:
```bash
python main.py --input data/audio.mp3 --language fr --task translate
```

3. Specifying a device (for Mac M-series processors):
```bash
python main.py --input data/audio.mp3 --language en --task transcribe --device mps
```

## Supported Audio Formats

- MP3
- WAV
- M4A (automatically converted to MP3 using FFmpeg)
- MOV (automatically converted to MP3 using FFmpeg)

## Output

The tool generates two output files:
1. A VTT file containing timestamped transcriptions
2. A plain text file with the complete transcription

Output files are saved in the same directory as the input file by default.

## Performance Notes

- For Mac users with M-series processors, using the `mps` device provides significant speed improvements compared to CPU processing
- For Windows/Linux users with NVIDIA GPUs, the `cuda` device will be automatically selected when available
- The tool processes audio in chunks of 30 seconds by default for optimal memory usage

## Troubleshooting

1. If you get FFmpeg-related errors:
   - Ensure FFmpeg is installed on your system
   - On Mac: `brew install ffmpeg`
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On Windows: Download from the official FFmpeg website

2. For Mac M-series users:
   - If you experience issues with MPS device, try falling back to CPU with `--device cpu`
