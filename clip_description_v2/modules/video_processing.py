import os
import json
import torch
import whisper
import torchaudio
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoModel, AutoTokenizer

from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_scene_timestamps(video_path, threshold=30.0, min_scene_len=1):
    """
    Extracts scene transition timestamps from a video using PySceneDetect.

    :param video_path: Path to the input video file.
    :param threshold: Threshold for ContentDetector (higher = fewer scenes).
    :param min_scene_len: Minimum scene length in seconds.
    :return: List of timestamps for scene transitions.
    """
    # Initialize VideoManager and SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Add ContentDetector for scene detection with custom threshold
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start the video manager and detect scenes
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get the list of scenes
    scene_list = scene_manager.get_scene_list()
    timestamps = []

    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        # Only include scenes longer than min_scene_len
        if end - start >= min_scene_len:
            timestamps.append((start, end))

    return timestamps

# split_video
def split_video_by_timestamps(input_video_path, timestamps, output_folder):
    """
    MP4 영상을 JSON 데이터에 나와 있는 시간대로 분할.

    Args:
        input_video_path (str): 입력 동영상 경로.
        timestamps (list of dict): 클립의 시작, 종료 시간 정보가 포함된 리스트.
        output_folder (str): 분할된 클립을 저장할 폴더 경로.
    """
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 동영상 로드
    video = VideoFileClip(input_video_path)

    for i, segment in enumerate(timestamps):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        # 클립 생성
        clip = video.subclipped(start_time, end_time)
        
        # 파일 이름 설정
        output_path = os.path.join(output_folder, f"clip_{i+1:03d}.mp4")
        
        # 클립 저장
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Clip {i+1} saved: {output_path}")

    # 동영상 객체 해제
    video.close()
    
def process_video(video_path, output_json_path):
    """
    Process the video to extract scene timestamps and transcribe each scene's audio.

    :param video_path: Path to the input video file.
    :param output_json_path: Path to save the output JSON file.
    """
    # Load Whisper model
    whisper_model = whisper.load_model("large")

    # Extract scene timestamps
    scene_timestamps = extract_scene_timestamps(video_path)

    # Prepare results
    results = []

    clip_id = 1  # Initialize clip ID

    for start, end in scene_timestamps:
        print(f"Processing scene from {start:.2f}s to {end:.2f}s...")
        text = transcribe_audio(video_path, start, end, whisper_model)
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
    