import os

def write_vtt_file(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, chunk in enumerate(chunks, start=1):
            start_time = format_timestamp(chunk['timestamp'][0])
            end_time = format_timestamp(chunk['timestamp'][1])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{chunk['text'].strip()}\n\n")

def format_timestamp(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"

def format_time_delta(td):
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds:06d}"

def get_output_filenames(args):
    output_base, ext = os.path.splitext(args.output) if args.output else (os.path.splitext(args.input)[0], '')
    vtt_output = args.output if ext else f"{output_base}.vtt"
    txt_output = args.output if ext else f"{output_base}.txt"
    return vtt_output, txt_output

def write_output_files(result, vtt_output, txt_output):
    write_vtt_file(result['chunks'], vtt_output)
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write(result['text'])
