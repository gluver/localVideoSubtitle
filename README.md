# localVideoSubtitle

Simple python CLI to generate subtitles in batches for local videos with whisper model , based on the [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) project

# How to Install 

```shell
!pip install -q pipx
```
```shell
!pipx install insanely-fast-whisper
```

If you got errors `AssertionError: Torch not compiled with CUDA enabled error on Windows`, try to reinsatll pytorch with

```shell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Usage
The script will walk through the user give path for matched media files (audio/video) , for video mode, ffmpeg will be used for extract the audio as an extra step.
*Example*
```shell
python batch_transcribe --input-path <root_path_where_media_files_stored> --language russian --mode video
```
```shell
usage: batch_transcribe.py [-h] --input-path INPUT_PATH [--device-id DEVICE_ID] [--output-path OUTPUT_PATH] [--transcript-path TRANSCRIPT_PATH]
                           [--subtitle-path SUBTITLE_PATH] [-f {txt,vtt,srt}] [--model-name MODEL_NAME] [--task {transcribe,translate}] [--language LANGUAGE]
                           [--batch-size BATCH_SIZE] [--flash FLASH] [--timestamp {chunk,word}] [--hf_token HF_TOKEN] [--diarization_model DIARIZATION_MODEL]
                           [--mode {audio,video}]

batch transcribe videos

options:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH
                        Path to the video file(s) to be transcribed.
  --device-id DEVICE_ID
                        Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")
  --output-path OUTPUT_PATH
                        Path to save the audio extract output. (default: current work directory)
  --transcript-path TRANSCRIPT_PATH
                        Path to save the transcription output. (default: current work directory)
  --subtitle-path SUBTITLE_PATH
                        Path to save the transcription output. (default: current work directory)
  -f {txt,vtt,srt}, --subtitle-format {txt,vtt,srt}
                        Format of the output file (default: srt)
  --model-name MODEL_NAME
                        Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)
  --task {transcribe,translate}
                        Task to perform: transcribe or translate to another language. (default: transcribe)
  --language LANGUAGE   Language of the input audio. (default: "None" (Whisper auto-detects the language))
  --batch-size BATCH_SIZE
                        Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)
  --flash FLASH         Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)
  --timestamp {chunk,word}
                        Whisper supports both chunked as well as word level timestamps. (default: chunk)
  --hf_token HF_TOKEN   Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips
  --diarization_model DIARIZATION_MODEL
                        Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)
  --mode {audio,video}  Video mode will use ffmpeg for audio extraction first in the pipeline,audio mode pipeline begin with audio files
```
