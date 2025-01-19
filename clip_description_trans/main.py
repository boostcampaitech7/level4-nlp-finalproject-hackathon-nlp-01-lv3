import os
import json
import yaml
from tqdm import tqdm
from modules.model_utils import initialize_model
from modules.video_processing import process_video, split_video_by_timestamps
from googletrans import Translator

def main():
    # Load configuration from YAML file
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    
    video_dir = config["video_dir"]
    folder_name = config["output_folder"]
    model_path = config["model_path"]
    mm_llm_compress = config["mm_llm_compress"]
    max_num_frames = config["max_num_frames"]
    generation_config = config["generation_config"]
    prompts = config["prompts"]
    final_output_path = config["final_output"]
    
    # Output folder setup
    os.makedirs(folder_name, exist_ok=True)

    # Get video files
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    #5개만 우선 테스트해봅니다.
    #video_files = video_files[:5]    
    
    # Initialize model and tokenizer
    model, tokenizer, image_processor = initialize_model(model_path, mm_llm_compress)
    
    # Initialize Google Translator
    translator = Translator()
    
    # 최종 JSON 데이터 구조 생성
    final_json_data = {"video_clips_info": []}
    
    # Process video and save results
    for video_file in video_files:
        # 비디오 경로 및 출력 파일 경로 설정
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # 비디오 이름 추출
        output_json_path = f"{folder_name}/output_segments_script_{video_name}.json"
        
        # 동영상 처리
        process_video(video_path, output_json_path)
        
        with open(output_json_path, 'r') as file:
            json_data = json.load(file)
            
        output_folder = f"clip/output_clips_{video_name}"

        # 동영상 클립 분할 실행
        split_video_by_timestamps(video_path, json_data, output_folder)
        
        with open(output_json_path, 'r') as f:
            scripts = json.load(f)
    
        # Format scripts for prompt
        script_texts = {}
        for script in scripts:
            script_texts[f"clip_{int(script['clip']):03}"] = script_texts.get(f"clip_{int(script['clip']):03}", "") + \
                f"[{script['start']} - {script['end']}] {script['text']}\n"
        
        # 각 비디오 클립 처리
        clip_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".mp4")])
        outputs = {}
        
        for clip_file in tqdm(clip_files, desc=f"Processing video files"):
            video_clip_path = os.path.join(output_folder, clip_file)
            clip_name = os.path.splitext(clip_file)[0]

            # 클립 이름이 스크립트와 매칭될 경우
            if clip_name in script_texts:
                prompt = prompts["clip_description"]
                prompt += script_texts[clip_name]
                
                output = model.chat(
                    video_path=video_clip_path,
                    tokenizer=tokenizer,
                    user_prompt=prompt,
                    return_history=False,
                    max_num_frames=max_num_frames,
                    generation_config=generation_config
                )

                # Translate description to Korean
                translated_description = translator.translate(output, src="en", dest="ko").text

                outputs[clip_name] = {
                    'script_texts': script_texts[clip_name],
                    'output': output,
                    'trans_output':translated_description
                }
                
                # 최종 JSON 데이터에 추가
                for script in scripts:
                    if f"clip_{script['clip']:03}" == clip_name:
                        final_json_data["video_clips_info"].append({
                            "video_file": video_file,
                            "start_timestamp": script["start"],
                            "end_timestamp": script["end"],
                            "clip_description": output,
                            "translated_description": translated_description,
                            "script": script["text"]
                        })

    # 최종 JSON 파일 저장
    with open(final_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)

    print(f"All outputs have been saved to {final_output_path}.")


main()