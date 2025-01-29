import copy
import warnings

import numpy as np
import torch
from decord import VideoReader, cpu
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, sample_fps, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).numpy()
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time


def load_llava_video_model():
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device_map = "auto"  # 오류 떠서 주석처리함 {"": "cuda"}

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        torch_dtype="bfloat16",
        device_map=device_map,
        attn_implementation="sdpa",
    )  # Add any other thing you want to pass in llava_model_args

    model.eval()
    model = model.half()

    return tokenizer, model, image_processor, max_length


def get_video_and_input_ids(
    video_path,
    tokenizer,
    model,
    image_processor,
    max_frames_num=64,
    prompt="Please describe this video in detail.",
):
    """
    Please describe this video in detail.
    """
    video, frame_time, video_time = load_video(
        video_path, max_frames_num, 1, force_sample=True
    )

    video = (
        image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        .cuda()
        .to(torch.float16)
    )
    video = [video]

    conv_template = (
        "qwen_1_5"  # Make sure you use correct chat template for different models
    )
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to("cuda")
    )
    return video, input_ids
