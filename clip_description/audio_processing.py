import os
import torch
import torchaudio
import whisper

def convert_to_mono(wav_path, output_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save(output_path, waveform, sample_rate)
    return output_path

def transcribe_audio(video_path, start_time, end_time, model):
    temp_audio_path = "temp_audio.wav"
    temp_mono_audio_path = "temp_audio_mono.wav"
    ffmpeg_command = f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 2 -ss {start_time} -to {end_time} -y {temp_audio_path}"
    os.system(ffmpeg_command)
    convert_to_mono(temp_audio_path, temp_mono_audio_path)
    result = model.transcribe(temp_mono_audio_path, language='en')
    os.remove(temp_audio_path)
    os.remove(temp_mono_audio_path)
    return result.get("text", "").strip()