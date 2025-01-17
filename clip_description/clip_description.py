import os
import torch
import whisper
from video_processing import extract_scene_timestamps
from audio_processing import transcribe_audio
from model_handling import load_model_and_tokenizer, predict
from utils import save_json
from datetime import timedelta


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def process_videos(video_folder, output_json_path, model_path):
    """
    Process all videos in a folder and save combined results in a JSON file.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load models
    model, tokenizer = load_model_and_tokenizer(model_path, device, torch_type)
    whisper_model = whisper.load_model("turbo")

    # Output structure
    output_data = {"video_clips_info": []}

    # Process each video in the folder
    for video_file in os.listdir(video_folder):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video: {video_file}")

        # Extract scene timestamps
        timestamps = extract_scene_timestamps(video_path)

        for idx, (start, end) in enumerate(timestamps):
            print(f"Processing scene {idx + 1}: {format_timestamp(start)} - {format_timestamp(end)}")

            # Transcribe audio
            text = transcribe_audio(video_path, start, end, whisper_model)

            # Generate clip description
            prompt = f"Describe the video clip from {format_timestamp(start)} to {format_timestamp(end)}."
            with open(video_path, 'rb') as f:
                video_data = f.read()
            clip_description = predict(prompt, video_data, model, tokenizer, device, torch_type)

            # Add to output
            output_data["video_clips_info"].append({
                "video_file": video_file,
                "start_timestamp": format_timestamp(start),
                "end_timestamp": format_timestamp(end),
                "clip_description": clip_description,
                "script": text
            })

    # Save output to JSON
    save_json(output_data, output_json_path)
    print(f"All videos processed. Results saved to {output_json_path}")
