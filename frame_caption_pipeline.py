"""
frame_caption_pipeline.py

코드 구조:
1. config 파일 로드
2. 비디오 폴더로부터 프레임 추출하여 frames 폴더에 저장
3. frames 폴더들의 이미지들로 데이터셋 생성 후 저장 (datasets 폴더에 저장)
4. 데이터셋으로부터 프레임 캡션 생성 후 Json 파일로 저장
"""

from code.frame_caption_modules import frame_caption
from code.frame_utils import create_and_save_dataset, extract_frames_from_folder
from code.utils import load_config

if __name__ == "__main__":
    # config 파일 로드
    config = load_config("./config/fcs_config.yaml")

    video_folder = config["extract_frames"]["video_folder"]
    frame_rate = config["extract_frames"]["frame_rate"]
    frames_folder = config["data"]["frames_folder"]
    datasets_folder = config["data"]["datasets_folder"]
    datasets_name = config["data"]["datasets_name"]

    device = config["general"]["device"]
    output_folder = config["data"]["output_folder"]
    model_name = config["model"]["model_name"]
    caption_prompt = config["generation"]["prompt"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    batch_size = config["generation"]["batch_size"]
    use_datasets = config["generation"]["use_datasets"]
    frame_output_filename = config["data"]["frame_output_filename"]

    # 비디오 폴더로부터 프레임 추출하여 frames 폴더에 저장
    extract_frames_from_folder(video_folder, frames_folder, frame_rate)

    # frames 폴더들의 이미지들로 데이터셋 생성 후 저장 (datasets 폴더에 저장)
    create_and_save_dataset(frames_folder, datasets_folder, datasets_name)

    # 데이터셋으로부터 프레임 캡션 생성 후 Json 파일로 저장
    frame_caption(
        device,
        frames_folder,
        output_folder,
        datasets_folder,
        datasets_name,
        model_name,
        caption_prompt,
        max_new_tokens,
        batch_size,
        use_datasets,
        frame_output_filename,
    )
