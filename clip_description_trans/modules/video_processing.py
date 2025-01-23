import os
import json
import torch
import whisper
import torchaudio
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoModel, AutoTokenizer
from scenedetect import open_video, SceneManager

from modules.audio_processing import transcribe_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
import re
import cv2

def extract_scene_timestamps(video_path, threshold=30.0, min_scene_len=2):
    """
    Extracts scene transition timestamps from a video using PySceneDetect.

    :param video_path: Path to the input video file.
    :param threshold: Threshold for ContentDetector (higher = fewer scenes).
    :param min_scene_len: Minimum scene length in seconds.
    :return: List of timestamps for scene transitions.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    timestamps = []

    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        if end - start >= min_scene_len:
            timestamps.append((start, end))

    return timestamps

def save_timestamps_to_txt(input_dir, output_txt, threshold=30.0, min_scene_len=2):
    """
    Extracts scene timestamps for all videos in the input directory and saves them to a txt file.

    :param input_dir: Directory containing input videos.
    :param output_txt: Path to the output txt file.
    :param threshold: Threshold for ContentDetector.
    :param min_scene_len: Minimum scene length in seconds.
    """
    with open(output_txt, "w") as txt_file:
        for video_name in os.listdir(input_dir):
            if not video_name.endswith(".mp4"):
                continue

            video_path = os.path.join(input_dir, video_name)
            video_id = os.path.splitext(video_name)[0]      

            try:
                timestamps = extract_scene_timestamps(video_path, threshold, min_scene_len)

                # Merge consecutive timestamps to ensure no gaps
                merged_timestamps = []
                previous_end = 0.0

                for start, end in timestamps:
                    if start > previous_end:
                        merged_timestamps.append((previous_end, start))
                    merged_timestamps.append((start, end))
                    previous_end = end

                # Add final segment if there is a gap at the end
                total_duration = float(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) / cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

                if previous_end < total_duration:
                    merged_timestamps.append((previous_end, total_duration))

                # Write to txt file
                for start, end in merged_timestamps:
                    txt_file.write(f"{video_id}\t{start:.3f}\t{end:.3f}\n")

            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                
        #return merged_timestamps

def split_clips_from_txt(input_dir, output_dir, timestamp_txt):
    """
    Reads timestamps from a txt file and splits videos into clips based on those timestamps.

    :param input_dir: Directory containing input videos.
    :param output_dir: Directory to save output clips.
    :param timestamp_txt: Path to the txt file containing timestamps.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(timestamp_txt, "r") as txt_file:
        for line in txt_file:
            video_id, start, end = line.strip().split("\t")
            start = float(start)
            end = float(end)

            video_path = os.path.join(input_dir, f"{video_id}.mp4")
            output_clip_path = os.path.join(output_dir, f"{video_id}_clip_{start:.3f}-{end:.3f}.mp4")

            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found. Skipping...")
                continue

            try:
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))

                start_frame = int(start * fps)
                end_frame = int(end * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()
                print(f"Saved clip: {output_clip_path}")

            except Exception as e:
                print(f"Error processing clip {output_clip_path}: {e}")


def reduce_repeated_characters(text, max_repeats=5):
    # 정규식을 사용하여 반복되는 문자를 최대 max_repeats로 줄임
    return re.sub(r"(.)\1{" + str(max_repeats) + r",}", r"\1" * max_repeats, text)

def process_video(video_path, output_json_path,timestamp_txt):

    # Load Whisper model
    whisper_model = whisper.load_model("large-v3")

    # Extract scene timestamps
    
    timestamps = read_timestamps_from_txt(timestamp_txt)
    
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Prepare results
    results = []
    clip_id = 1  # Initialize clip ID
    
    for start, end in timestamps[video_id]:
        print(f"Processing scene from {start:.2f}s to {end:.2f}s...")
        text = transcribe_audio(video_path, start, end, whisper_model)
        text = reduce_repeated_characters(text)
        results.append({
            "clip": clip_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text
        })
        clip_id += 1  # Increment clip ID

    # Save results to JSON file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_json_path}")
    
def read_timestamps_from_txt(timestamp_txt):
    """
    Reads timestamps from a txt file and converts them into a dictionary.

    :param timestamp_txt: Path to the txt file containing timestamps.
    :return: Dictionary of video IDs and their timestamps.
    """
    timestamps_dict = {}

    with open(timestamp_txt, "r") as txt_file:
        for line in txt_file:
            video_id, start, end = line.strip().split("\t")
            start = float(start)
            end = float(end)

            if video_id not in timestamps_dict:
                timestamps_dict[video_id] = []

            timestamps_dict[video_id].append((start, end))

    return timestamps_dict