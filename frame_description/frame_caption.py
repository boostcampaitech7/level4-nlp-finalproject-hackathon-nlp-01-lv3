import os
import cv2
import json
import yaml
import torch
import datetime
from tqdm import tqdm
from PIL import Image
from googletrans import Translator
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# YAML 설정 파일 로드
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# 초 단위를 "HH:MM:SS.s" 형식으로 변환하는 함수
def seconds_to_hms_ms(seconds):
    is_negative = seconds < 0
    seconds = round(abs(seconds) * 2) / 2
    hms = str(datetime.timedelta(seconds=int(seconds)))
    fraction = ".5" if seconds % 1 == 0.5 else ".0"
    hms_ms = f"{hms}{fraction}"
    return f"-{hms_ms}" if is_negative else hms_ms

# 프레임 추출 함수
def extract_frames(video_path, output_dir, frame_rate):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 1
    frame_count = 0
    saved_count = 0

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            timestamp = round((frame_count / fps if fps > 0 else frame_count) * 2) / 2
            frame_filename = f"{video_id}_{timestamp:.1f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_id}: 추출된 프레임 수 = {saved_count}")

# main 함수
def main(config_path):
    # 설정 로드
    config = load_config(config_path)
    device = config["general"]["device"]
    video_folder = config["general"]["video_folder"]
    frames_folder = config["general"]["frames_folder"]
    output_folder = config["general"]["output_folder"]
    frame_rate = config["general"]["frame_rate"]

    # 모델 로드
    model_name = config["model"]["blip2_model_name"]
    torch_dtype = torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.float32
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
    ).to(device).eval()

    translator = Translator()
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    print(f"총 {len(video_files)}개의 비디오 파일을 발견했습니다.")

    # 프레임 추출
    for video_file in tqdm(video_files, desc="비디오 프레임 추출 중"):
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, frames_folder, frame_rate)

    # 프레임 처리
    frame_files = sorted(
        [f for f in os.listdir(frames_folder) if f.endswith(".jpg")],
        key=lambda x: float(x.rsplit("_", 1)[1].split(".")[0])  # timestamp 기준 정렬
    )
    print(f"총 {len(frame_files)}개의 프레임 파일을 발견했습니다.")

    results = []
    for frame_file in tqdm(frame_files, desc="프레임에 텍스트 생성 중"):
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)

        try:
            translation = translator.translate(caption, dest=config["translation"]["language"])
            translation = translation.text
        except Exception as e:
            print(f"번역 실패: {caption}. 오류: {e}")
            translation = ""

        video_id, timestamp = frame_file.rsplit("_", 1)
        timestamp = seconds_to_hms_ms(float(timestamp.split(".")[0]))
        results.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "frame_image_path": frame_path,
            "caption": caption,
            "caption_ko": translation,
        })

    # JSON 저장
    output_json_path = os.path.join(output_folder, "frame_output_v.json")
    os.makedirs(output_folder, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"텍스트 생성 완료. 결과가 {output_json_path}에 저장되었습니다.")

if __name__ == "__main__":
    main("fc_config.yaml")
