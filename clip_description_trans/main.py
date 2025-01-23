import os
import json
import yaml
from tqdm import tqdm
from modules.model_utils import initialize_model
from modules.video_processing import *
from googletrans import Translator
import torch

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
    # Load configuration from YAML file
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    
    video_dir = config["video_dir"]
    output_directory = config["clip_output_directory"] #full_clip저장할곳
    timestamp_file = config["timestamp_file"]
    folder_name = config["output_folder"] #script
    model_path = config["model_path"]
    mm_llm_compress = config["mm_llm_compress"]
    max_num_frames = config["max_num_frames"]
    
    generation_config = config["generation_config"]
    prompt_config = config["prompt"]
    final_output_path = config["final_output"]
    
    # Output folder setup
    os.makedirs(folder_name, exist_ok=True)

    # Get video files
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    print(video_files)
    save_timestamps_to_txt(video_dir, timestamp_file, threshold=30.0, min_scene_len=2)
    # 동영상 클립 분할 실행
    split_clips_from_txt(video_dir, output_directory, timestamp_file)
    #5개만 우선 테스트해봅니다.
    #video_files = video_files[:2]    
    
    # Initialize model and tokenizer
    model, tokenizer, image_processor = initialize_model(model_path, mm_llm_compress)
    
    # Initialize Google Translator
    translator = Translator()
    
    # 최종 JSON 데이터 구조 생성
    final_json_data = {"video_clips_info": []}
    
    # Process video and save results
    for video_file in tqdm(video_files):
        print(f"Processing video: {video_file}")
        # 비디오 경로 및 출력 파일 경로 설정
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # 비디오 이름 추출
        output_json_path = f"{folder_name}/output_segments_script_{video_name}.json"
        
        # 동영상 처리
        process_video(video_path, output_json_path,timestamp_file)
        
            
        #output_folder = f"clip/output_clips_{video_name}"

        #input_dir, output_dir, timestamp_txt
        
        with open(output_json_path, 'r') as f:
            scripts = json.load(f)
    
        # Format scripts for prompt
        script_texts = {}
        for script in scripts:
            print(f"Scripts: {scripts}")
            script_texts[f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}"] = script_texts.get(f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}", "") + f"{script['text']}\n"
                #f"[{script['start']} - {script['end']}] {script['text']}\n"


        # 각 비디오 클립 처리
        clip_files = sorted([f for f in os.listdir(output_directory) if f.endswith(".mp4")])
        outputs = {}
        
        for clip_file in tqdm(clip_files, desc=f"Processing video files"):
            video_clip_path = os.path.join(output_directory, clip_file)
            clip_name = os.path.splitext(clip_file)[0]
            print("============================================================================")
            print(f"clip_name: {clip_name}, script_texts keys: {list(script_texts.keys())}")


            # 클립 이름이 스크립트와 매칭될 경우
            if clip_name in script_texts:
                clip_prompt = prompt_config["clip_prompt_template"]
                clip_prompt += script_texts[clip_name]
                
                output = model.chat(
                    video_path=video_clip_path,
                    tokenizer=tokenizer,
                    user_prompt=clip_prompt,
                    return_history=False,
                    max_num_frames=max_num_frames,
                    generation_config=generation_config
                )
                print(f"Output: {output}")

                # mp4 지우기 
                if video_file.endswith(".mp4"):
                    vidoe_id = video_file[:-4]
                    
                #clip_number = clip_name.split('_')[-1]  # Assuming clip_name includes "_<clip_number>"
                #clip_id = f"{vidoe_id}_{clip_number}"

                # Translate description to Korean
                translated_description = translator.translate(output, src="en", dest="ko").text
                print(f"Translated: {translated_description}")

                outputs[clip_name] = {
                    'script_texts': script_texts[clip_name],
                    'output': output,
                    'trans_output':translated_description
                }
                
                # mp4 지우기 
                if video_file.endswith(".mp4"):
                    vidoe_id = video_file[:-4]
            
                # 최종 JSON 데이터에 추가
                for script in scripts:
                    if f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}" == clip_name:
                        final_json_data["video_clips_info"].append({
                            "video_id": vidoe_id,
                            "start_timestamp": script["start"],
                            "end_timestamp": script["end"],
                            "clip_id":script["clip"],
                            "clip_description": output,
                            "clip_description_ko": translated_description,
                            "script": script["text"]
                        })
                        
                        
    # 정렬 및 저장장
    sorted_clips_info  = sorted(final_json_data['video_clips_info'],key=lambda x: (x['video_id'], x['start_timestamp']),reverse=False)
    final_json_data['video_clips_info'] = sorted_clips_info

    with open(final_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"All outputs have been saved to {final_output_path}.")


main()