'''
영상 로드, 장면 탐지, 클립 분할과 관련된 함수들.
'''
import os
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_scene_timestamps(video_path, threshold=30.0, min_scene_len=1):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    timestamps = [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_list if scene[1].get_seconds() - scene[0].get_seconds() >= min_scene_len
    ]
    return timestamps

def split_video_by_timestamps(input_video_path, timestamps, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = VideoFileClip(input_video_path)

    for i, segment in enumerate(timestamps):
        start_time, end_time = segment
        clip = video.subclip(start_time, end_time)
        output_path = os.path.join(output_folder, f"clip_{i+1:03d}.mp4")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Clip {i+1} saved: {output_path}")
    video.close()
