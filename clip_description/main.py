import os
from video_script_json import process_videos

if __name__ == "__main__":
    video_folder = "/data/ephemeral/home/sujin/sample_video"
    output_json_path = "processed_videos.json" 
    model_path = "THUDM/cogvlm2-llama3-caption" # 모델 선택 
    
    if not os.path.exists(video_folder):
        print(f"Video folder '{video_folder}' not found!")
    else:
        process_videos(video_folder, output_json_path, model_path)