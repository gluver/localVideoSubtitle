import sys
import os
import argparse
import subprocess
import json


parser= argparse.ArgumentParser(description="batch transcribe videos")

parser.add_argument(
    "--input-path",
    required=True,
    type=str,
    help="Path to the video file(s) to be transcribed.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
)
parser.add_argument(
    "--output-path",
    required=False,
    default=f".{os.sep}output{os.sep}temp",
    type=str,
    help="Path to save the audio extract output. (default: current work directory)",
)

parser.add_argument(
    "--transcript-path",
    required=False,
    default=f".{os.sep}output{os.sep}transcript",
    type=str,
    help="Path to save the transcription output. (default: current work directory)",
)


parser.add_argument(
    "--subtitle-path",
    required=False,
    default=f".{os.sep}output{os.sep}subtitle",
    type=str,
    help="Path to save the transcription output. (default: current work directory)",
)

parser.add_argument(
    "-f", "--subtitle-format",
    default="srt", 
    help="Format of the output file (default: srt)", choices=["txt", "vtt", "srt"])

parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf_token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)

parser.add_argument(
    "--mode",
    default="audio",
    type=str,
    help="Video mode will use ffmpeg for audio extraction first in the pipeline,audio mode pipeline begin with audio files",
    choices=["audio", "video"]
)

def extract_audio(filename,audio_output_path):
    filename_base= os.path.splitext(os.path.split(filename)[-1])[0]
    audio_name=os.path.join(audio_output_path,filename_base+".wav")
    os.makedirs(audio_output_path,exist_ok="Ture")
    result=subprocess.run(['ffmpeg','-i',filename,'-map','0:a:0','-vn','-c','copy',audio_name])
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    return audio_name

def generate_transcript(audio_filename,args):
    filename_base= os.path.splitext(os.path.split(audio_filename)[-1])[0]
    os.makedirs(args.transcript_path,exist_ok="Ture")
    transcript_name=os.path.join(args.transcript_path,filename_base+".json")
    result=subprocess.run(["insanely-fast-whisper","--file-name",audio_filename,
                           "--transcript-path",transcript_name,
                           "--language",args.language,
                           "--model-name",args.model_name,
                           "--task",args.task,
                           "--timestamp",args.timestamp,
                           ])
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    return transcript_name

#Formatting json output to subtitle needed form
class TxtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        return f"{text}\n"


class SrtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"


class VttFormatter:
    @classmethod
    def preamble(cls):
        return "WEBVTT\n\n"

    @classmethod
    def format_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"


def convert(input_path, output_format, output_dir):
    filename_base= os.path.splitext(os.path.split(input_path)[-1])[0]
    with open(input_path, 'r',encoding="utf-8") as file:
        data = json.load(file)

    formatter_class = {
        'srt': SrtFormatter,
        'vtt': VttFormatter,
        'txt': TxtFormatter
    }.get(output_format)

    string = formatter_class.preamble()
    for index, chunk in enumerate(data['chunks'], 1):
        entry = formatter_class.format_chunk(chunk, index)


        string += entry
    os.makedirs(output_dir,exist_ok="Ture")
    with open(os.path.join(output_dir, f"{filename_base}.{output_format}"), 'w', encoding='utf-8') as file:
        file.write(string)



def from_video(args):
  
    # extract audio
    for dirpath,dirnames,filenames in os.walk(args.input_path):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        for filename in filenames:
            if os.path.splitext(filename)[1] in video_extensions:
                filepath=os.path.join(dirpath,filename)
                relative_path = dirpath.replace(args.input_path, '').lstrip(os.sep)
                audio_output_path= os.path.join(args.output_path,relative_path)
                #extract and return the audio file full path
                audio_name=extract_audio(filepath,audio_output_path)
                #inference and generate transcript json with openai whisper model
                transcript_name=generate_transcript(audio_name,args)
                #covert the json to subtitle format
                convert(transcript_name,args.subtitle_format,args.subtitle_path)
        # extract_audio(video_file,args.transcript_path)
                
def from_audio(args):
    for dirpath,dirnames,filenames in os.walk(args.input_path):
        audio_extensions = ('.mp3', '.wav')
        for filename in filenames:
            if os.path.splitext(filename)[1] in audio_extensions:
                audio_file_path=os.path.join(dirpath,filename)
                #inference and generate transcript json with openai whisper model
                transcript_name=generate_transcript(audio_file_path,args)
                #covert the json to subtitle format
                convert(transcript_name,args.subtitle_format,args.subtitle_path)
        # extract_audio(video_file,args.transcript_path)

if __name__ == "__main__":
    args=parser.parse_args()
    if args.mode=='audio':
        from_audio(args)
    else:
        from_video(args)
    
