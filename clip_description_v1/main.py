from modules.video_processing import process_video, split_video_by_timestamps
import os
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def main():
    # Output JSON path
    folder_name = 'scene_script'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 비디오 파일 리스트 가져오기
    video_dir = "/workspace/own_dataset_video/"
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    
    #=====================모델 관련 설정=======================
    # Load the model and tokenizer
    model_path = 'OpenGVLab/VideoChat-Flash-Qwen2-7B_res224'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    image_processor = model.get_vision_tower().image_processor
    
    # Configure the model
    mm_llm_compress = False
    if mm_llm_compress:
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        model.config.mm_llm_compress = True
        
    # Evaluation configuration
    max_num_frames = 96
    generation_config = dict(
        do_sample=False,
        max_new_tokens=1024,
        num_beams=1
    )
    #========================================================

    # 최종 JSON 데이터 구조 생성
    final_json_data = {"video_clips_info": []}
    
    # Process video and save results
    for video_file in video_files:
        # 비디오 경로 및 출력 파일 경로 설정
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # 비디오 이름 추출
        output_json_path = f"{folder_name}/output_segments_script_{video_name}.json"
        output_folder = f"output_clips_{video_name}"
        
        # 동영상 처리
        process_video(video_path, output_json_path)
        
        with open(output_json_path, 'r') as file:
            scripts = json.load(file)

        # 동영상 클립 분할 실행
        split_video_by_timestamps(video_path, scripts, output_folder)
        
        # Format scripts for prompt
        script_texts = {
            f"clip_{int(script['clip']):03}": f"[{script['start']} - {script['end']}] {script['text']}\n"
            for script in scripts
        }
                
        # 초기 프롬프트 설정
        init_prompt = "The video provides an overall context. Please describe each subsequent video clip based on its content and how it connects to the rest of the video."
        init_output, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=init_prompt, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

        # 각 비디오 클립 처리
        clip_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".mp4")])
        for clip_file in tqdm(clip_files, desc=f"Processing clips for {video_file}"):
            video_clip_path = os.path.join(output_folder, clip_file)
            clip_name = os.path.splitext(clip_file)[0]

            # 클립 이름이 스크립트와 매칭될 경우
            if clip_name in script_texts:
                prompt = f"Describe this video in detail, considering the scripts provided below:\n{script_texts[clip_name]}"

                output, chat_history = model.chat(
                    video_path=video_clip_path,
                    tokenizer=tokenizer,
                    user_prompt=prompt,
                    chat_history=chat_history,
                    return_history=True,
                    max_num_frames=max_num_frames,
                    generation_config=generation_config
                )

                # 최종 JSON 데이터에 추가
                for script in scripts:
                    if f"clip_{script['clip']:03}" == clip_name:
                        final_json_data["video_clips_info"].append({
                            "video_file": video_file,
                            "start_timestamp": script["start"],
                            "end_timestamp": script["end"],
                            "clip_description": output,
                            "script": script["text"]
                        })

    # 최종 JSON 파일 저장
    final_output_path = "final_output_2.json"
    with open(final_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)

    print(f"All outputs have been saved to {final_output_path}.")


main()