import os
import subprocess

def ensure_compatible_audio(input_path: str) -> str:
    """
    Ensures the audio file is in a compatible format.
    Converts m4a to mp3 if necessary.
    Returns the path to the compatible audio file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in ['.wav', '.mp3']:
        return input_path  # These formats are directly compatible
    elif file_extension == '.m4a':
        return convert_m4a_to_mp3(input_path)
    else:
        raise ValueError(f"Unsupported audio format: {file_extension}")

def convert_m4a_to_mp3(input_path: str) -> str:
    """
    Converts m4a file to mp3 format using ffmpeg with high-quality settings.
    """
    output_path = os.path.splitext(input_path)[0] + ".mp3"

    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Automatically overwrite output files
        "-i", input_path,
        "-ac", "1",  # Convert to mono
        "-ar", "16000",  # Set sample rate to 16kHz
        "-codec:a", "libmp3lame",
        "-b:a", "320k",  # Set bitrate to 320 kbps for high quality
        "-hide_banner",
        "-loglevel", "error",
        output_path
    ]

    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        subprocess.run(ffmpeg_command, check=True)
        return output_path
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffmpeg conversion from m4a to mp3 failed: {error}") from error
    except FileNotFoundError:
        raise EnvironmentError("ffmpeg is not installed or not found in the system's PATH")