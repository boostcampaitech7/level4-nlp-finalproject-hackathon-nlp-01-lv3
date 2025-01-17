'''
모델과 토크나이저 로딩, 예측 함수 구현
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from decord import cpu, VideoReader, bridge
import numpy as np 
import io

def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

def load_model_and_tokenizer(model_path, device, torch_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_type, trust_remote_code=True
    ).eval().to(device)
    return model, tokenizer

def predict(prompt, video_data, model, tokenizer, device, torch_type, strategy='chat'):
    video = load_video(video_data, strategy=strategy)
    history = []
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=prompt,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch_type)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": 0.1,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
