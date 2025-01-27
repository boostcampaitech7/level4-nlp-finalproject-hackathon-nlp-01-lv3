"""
scene_caption_modules.py

함수 목록:
1. initialize_model
2. scene_caption_InternVideo2

추후 구현 예정:
3. scene_caption
"""

import json
import os

import decord
import torch
from audio_utils import transcribe_audio
from googletrans import Translator
from specific_model_utils.InternVideo2_utils import load_video
from transformers import AutoModel, AutoTokenizer

decord.bridge.set_bridge("torch")


def initialize_model(model_path="OpenGVLab/InternVideo2-Chat-8B"):
    """
    model_path를 받아 모델과 tokenizer를 반환하는 함수

    Args:
        model_path (str): 모델 경로

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    return model, tokenizer


def scene_caption_InternVideo2(
    model_path,
    scene_path,
    prompt,
    translator,
    generation_config,
    use_audio_for_prompt,
    scene_info_json_file_path=None,  # 오디오 스크립트 정보 포함
):
    """
    1개의 scene에 대한 캡션을 InternVideo2 모델을 사용하여 생성하는 함수

    Args:
        model_path (str): 모델 경로
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        translator (googletrans.Translator): 번역기
        generation_config (dict): 생성 설정
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        result (dict): 캡션 결과
    """
    model, tokenizer = initialize_model(model_path)
    translator = Translator()

    scene_tensor = load_video(scene_path, num_segments=8, return_msg=False)
    scene_tensor = scene_tensor.to(model.device)

    # scene_name 추출 (audio_name이랑 같음 - {video_id}_{start:.3f}_{end:.3f}_{i + 1:03d})
    scene_name = os.path.basename(scene_path).split(".")[0]
    video_id, start, end, scene_id = scene_name.split("_")

    if use_audio_for_prompt:
        if scene_info_json_file_path:
            audio_path = os.path.join(scene_info_json_file_path, scene_name + ".wav")
            with open(audio_path, "r") as f:
                audio_text = json.load(f)[video_id][int(scene_id) - 1]["audio_text"]
        else:
            # STT 모델인 Whisper를 불러옴
            import whisper

            whisper_model = whisper.load_model("large-v3")
            audio_text = transcribe_audio(scene_path, whisper_model)

    prompt = (
        prompt["clip_prompt_template"] + f"\n[script]: {audio_text}"
        if use_audio_for_prompt
        else prompt["clip_prompt_template"]
    )

    chat_history = []
    response, chat_history = model.chat(
        tokenizer,
        "",
        prompt,
        media_type="video",
        media_tensor=scene_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config=generation_config,
    )

    translated_description = translator.translate(response, src="en", dest="ko").text

    result = {
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "clip_id": f"{video_id}_{start}_{end}_{scene_id}",
        "caption": response,
        "caption_ko": translated_description,
    }
    return result


if __name__ == "__main__":
    pass
