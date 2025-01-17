import os
import cv2
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoModel, AutoTokenizer
from googletrans import Translator
import datetime
import torchvision.transforms as T
from decord import VideoReader, cpu

class VideoFrameCaptionGenerator:
    def __init__(self, video_folder, frames_folder, output_folder, model_name, model_type="blip2", frame_rate=2):
        self.video_folder = video_folder
        self.frames_folder = frames_folder
        self.output_folder = output_folder
        self.frame_rate = frame_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_type = model_type
        self.processor = None
        self.model = None
        self.translator = Translator()

        self._load_model()

    def _load_model(self):
        print(f"[INFO] Loading model: {self.model_name} ({self.model_type})")
        if self.model_type == "blip2":
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            ).eval().to(self.device)
        elif self.model_type == "custom":
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    @staticmethod
    def seconds_to_hms_ms(seconds):
        milliseconds = int((seconds % 1) * 1000)
        hms = str(datetime.timedelta(seconds=int(seconds)))
        return f"{hms}.{milliseconds:03d}"

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / self.frame_rate) if fps > 0 else 1
        frame_count = 0
        saved_count = 0

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = self.frames_folder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count
                frame_filename = f"{video_id}_{timestamp:.3f}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"{video_id}: Extracted {saved_count} frames")

    @staticmethod
    def extract_video_id_and_timestamp(file_name):
        try:
            file_base = os.path.splitext(file_name)[0]
            video_id, timestamp_str = file_base.rsplit('_', 1)
            timestamp = float(timestamp_str)
            return video_id, timestamp
        except (IndexError, ValueError):
            return '', float('inf')

    def generate_captions(self):
        frame_files = sorted(
            [f for f in os.listdir(self.frames_folder) if f.endswith('.jpg')],
            key=lambda x: self.extract_video_id_and_timestamp(x)
        )

        print(f"Found {len(frame_files)} frame files")
        results = []

        for frame_file in tqdm(frame_files, desc="Generating captions for frames"):
            frame_path = os.path.join(self.frames_folder, frame_file)
            image = Image.open(frame_path).convert('RGB')

            if self.model_type == "blip2":
                inputs = self.processor(image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    out = self.model.generate(**inputs)

                caption = self.processor.decode(out[0], skip_special_tokens=True)

            elif self.model_type == "custom":
                transform = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
                image_tensor = transform(image).unsqueeze(0).to(self.device)

                question = "<image>\nPlease describe the image in detail."
                with torch.no_grad():
                    caption = self.model.chat(self.tokenizer, image_tensor, question, dict(max_new_tokens=1024))

            try:
                translation = self.translator.translate(caption, dest='ko')
                caption = translation.text
            except Exception as e:
                print(f"Translation failed: {caption}. Error: {e}")

            frame_id = os.path.splitext(frame_file)[0]
            try:
                video_id, timestamp = frame_id.rsplit('_', 1)
                timestamp_hms = self.seconds_to_hms_ms(float(timestamp))
            except ValueError:
                video_id = frame_id
                timestamp_hms = "00:00:00.000"

            result = {
                "video_id": video_id,
                "timestamp": timestamp_hms,
                "frame_image_path": frame_path,
                "caption": caption
            }
            results.append(result)

        output_json_path = os.path.join(self.output_folder, "frame_captions.json")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Caption generation completed. Results saved to {output_json_path}")

    def process_videos(self):
        video_files = [f for f in os.listdir(self.video_folder) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} video files")

        for video_file in tqdm(video_files, desc="Extracting frames from videos"):
            video_path = os.path.join(self.video_folder, video_file)
            self.extract_frames(video_path)

        self.generate_captions()

# Example usage
if __name__ == "__main__":
    generator = VideoFrameCaptionGenerator(
        video_folder='/data/ephemeral/home/yunseo_final/dataset/dataset_video_sample',
        frames_folder='/data/ephemeral/home/yunseo_final/dataset/frames',
        output_folder='/data/ephemeral/home/yunseo_final/dataset/output_frame_description',
        model_name="Salesforce/instructblip-flan-t5-xxl",
        model_type="blip2",
        frame_rate=2
    )
    generator.process_videos()
