"""
캡셔닝할 때 사용하는 함수들을 정의한 파일입니다.

함수 목록:

"""

import whisper

from .LlavaVideo_utils import (
    get_video_and_input_ids,
    load_llava_video_model,
)
from .vtt_service_utils import transcribe_audio, translate_caption


def initialize_llava_video_model():
    """
    LLaVA-Video 모델을 초기화하여 tokenizer, model, image_processor를 반환하는 함수

    Returns:
        tokenizer, model, image_processor
    """
    tokenizer, model, image_processor, _ = load_llava_video_model()
    return tokenizer, model, image_processor


def initialize_whisper():
    whisper_model = whisper.load_model("large-v3")
    return whisper_model


def single_scene_caption_LlavaVideo(
    model,
    tokenizer,
    image_processor,
    scene_path,
    prompt,
    max_new_tokens,
    max_num_frames,
    enable_audio_text,
    whisper_model,
    mono_audio_path,
    translator,
):
    """
    1개의 scene에 대한 캡션을 LlavaVideo 모델을 사용하여 생성하는 함수

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): tokenizer
        image_processor: 이미지 처리기
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        max_new_tokens (int): 최대 토큰 수
        max_num_frames (int): 최대 프레임 수
        enable_audio_text (bool): 오디오 텍스트 사용 여부
        whisper_model (whisper.Whisper): STT 모델
        mono_audio_folder (str): 모노 오디오 폴더 경로
        translator (googletrans.Translator or deepl.Translator): 번역기 객체

    Returns:
        response, translated_description (str, str): 생성된 영어 캡션, 번역된 캡션
    """
    # scene_name 추출 (audio_name이랑 같음)
    # audio_name = scene_name = os.path.basename(scene_path)[: -len(".mp4")]

    if enable_audio_text:
        # STT 모델인 Whisper을 사용하여 오디오 텍스트 추출
        audio_text = transcribe_audio(mono_audio_path, whisper_model)

        # 프롬프트에 오디오 텍스트를 넣어주어 오디오를 반영하여 캡션 생성
        prompt += f"\n[SCRIPT]: {audio_text}[EOS]"

        print(prompt)

    video, input_ids = get_video_and_input_ids(
        scene_path, tokenizer, model, image_processor, max_num_frames, prompt
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long().to("cuda")

    cont = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    translated_description = translate_caption(response, translator, target_lang="ko")

    return response, translated_description
