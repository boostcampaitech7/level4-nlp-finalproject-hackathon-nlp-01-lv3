import bisect
import json
import os

import clip
import numpy as np
import torch
import torch.nn as nn
import yaml
import spacy
from lavis.models import load_model_and_preprocess
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# from rank_bm25 import BM25Okapi
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


##############################################
# Helper Function: 이미지 파일 로딩 및 전처리
##############################################
class ImageDataset(Dataset):
    def __init__(self, image_paths: list, preprocess, convert_mode: str = None):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.convert_mode = convert_mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path)
            if self.convert_mode:
                img = img.convert(self.convert_mode)
        except Exception as e:
            print(f"이미지 로딩 오류 ({path}): {e}")
            img = Image.new("RGB", (224, 224))
        processed = self.preprocess(img)
        return processed, path


##############################################
# 공통 부모 클래스: BGERetrieval
# - config 파일로부터 모델, 토크나이저, JSON 데이터, 임베딩 파일 경로 등을 로드합니다.
# - 텍스트 리스트(self.texts)에 대해 배치 단위로 임베딩을 수행하고, 코사인 유사도를 계산하는 공통 기능을 제공합니다.
##############################################
class BGERetrieval:
    def __init__(
        self, config_path: str, config_section: str, json_key: str, text_field: str
    ):
        """
        Args:
            config_path (str): 설정 파일 경로.
            config_section (str): 사용하고자 하는 config 내 섹션 이름 (예: "bge-scene" 또는 "bge-script").
            json_key (str): JSON 파일에서 대상 데이터 리스트의 key (예: "scenes" 또는 "scripts").
            text_field (str): 각 항목에서 텍스트로 사용할 필드 이름 (예: "caption" 또는 "summary").
        """
        full_config = self._load_config(config_path)
        if config_section not in full_config:
            raise KeyError(
                f"'{config_section}' 설정이 {config_path}에 존재하지 않습니다."
            )
        self.config = full_config[config_section]
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # BGE 모델 및 토크나이저 로드
        self.model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # JSON 파일 로드: json_key에 해당하는 항목 리스트를 self.data_info로 저장
        self.json_file = self.config["json_file"]
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data_info = data.get(json_key, [])
        self.texts = [item[text_field] for item in self.data_info]

        # 임베딩 파일 저장/로딩
        self.embedding_file = self.config["output_file"]
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        if os.path.exists(self.embedding_file):
            self._load_embeddings(self.embedding_file)
        else:
            self.embeddings = self._encode_texts(self.texts)
            self._save_embeddings(self.embedding_file)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _encode_texts(
        self, texts: list, batch_size: int = 1024, disable: bool = False
    ) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc="Encoding texts", disable=disable
        ):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 일반적으로 CLS 토큰(첫 번째 토큰) 벡터를 사용
                embeds = outputs.last_hidden_state[:, 0, :]
                # L2 Normalization
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)

            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0).to(self.device)

    def _save_embeddings(self, file_path: str) -> None:
        torch.save(
            {"data_info": self.data_info, "features": self.embeddings},
            file_path,
        )
        print(f"[BGE] 임베딩을 {file_path}에 저장했습니다.")

    def _load_embeddings(self, file_path: str) -> None:
        data = torch.load(file_path)
        self.data_info = data["data_info"]
        self.embeddings = data["features"]
        print(f"[BGE] 임베딩을 {file_path}에서 불러왔습니다.")

    def _compute_similarity(
        self, query_vec: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        query_vec = query_vec.to(features.device).float()
        features = features.float()
        return (features @ query_vec.T).squeeze(1)

    # retrieve()는 하위 클래스에서 구체적인 출력 포맷에 맞게 오버라이드합니다.
    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        raise NotImplementedError("하위 클래스에서 retrieve() 메서드를 구현하세요.")


##############################################
# SCENERetrieval: JSON의 scene 정보("scenes" 리스트, 각 항목의 캡션은 "caption") 기반 retrieval
##############################################
class SCENERetrieval(BGERetrieval):
    def __init__(self, config_path: str = "config/merged_config.yaml"):
        # bge-scene 섹션, JSON 내 "scenes" 항목, 텍스트 필드는 "caption" 사용
        super().__init__(
            config_path,
            config_section="bge-scene",
            json_key="scenes",
            text_field="caption",
        )
        self.scene_index = self._build_scene_index_from_json(self.json_file)

    def _build_scene_index_from_json(self, json_file: str) -> dict:
        """
        JSON 파일 내의 scene 정보를 읽어, video_id별로 start_time 기준으로 정렬된 index를 구성합니다.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes = data.get("scenes", [])
        scene_index = {}
        for scene in scenes:
            video_id = scene.get("video_id")
            if not video_id:
                continue
            try:
                start_time = float(scene["start_time"])
                end_time = float(scene["end_time"])
            except Exception as e:
                print(f"scene 파싱 오류: {e}")
                continue

            # 초기 score는 0.0
            scene_entry = {
                "scene_id": scene.get("scene_id"),
                "start_time": start_time,
                "end_time": end_time,
                "caption": scene.get("caption"),
                "score": 0.0,
            }
            if video_id not in scene_index:
                scene_index[video_id] = {"starts": [], "scenes": []}
            scene_index[video_id]["starts"].append(start_time)
            scene_index[video_id]["scenes"].append(scene_entry)

        # start_time 기준 정렬
        for vid in scene_index:
            combined = list(zip(scene_index[vid]["starts"], scene_index[vid]["scenes"]))
            combined.sort(key=lambda x: x[0])
            scene_index[vid]["starts"] = [item[0] for item in combined]
            scene_index[vid]["scenes"] = [item[1] for item in combined]
        return scene_index

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        사용자 쿼리에 대해 scene 임베딩 유사도를 계산하여 상위 top_k 결과를 반환합니다.
        """
        query_vec = self._encode_texts([user_query], disable=True)
        scores = self._compute_similarity(query_vec, self.embeddings)
        scores_np = scores.cpu().numpy()
        score_mapping = {
            self.data_info[i]["scene_id"]: float(score)
            for i, score in enumerate(scores_np)
        }
        top_idxs = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            info = self.data_info[idx]
            scene_id = info["scene_id"]
            results.append(
                {
                    "rank": rank,
                    "video_id": info.get("video_id"),
                    "scene_start_time": info["start_time"],
                    "scene_end_time": info["end_time"],
                    "scene_description": info["caption"],
                    "scene_id": scene_id,
                    "score": score_mapping[scene_id],
                }
            )
        return results


##############################################
# SCRIPTRetrieval: JSON의 script 정보("scripts" 리스트, 각 항목의 요약은 "summary") 기반 retrieval
##############################################
class SCRIPTRetrieval(BGERetrieval):
    def __init__(self, config_path: str = "config/merged_config.yaml"):
        # bge-script 섹션, JSON 내 "scripts" 항목, 텍스트 필드는 "summary" 사용
        super().__init__(
            config_path,
            config_section="bge-script",
            json_key="scripts",
            text_field="summary",
        )
        # script 전용 인덱스 구축 (필요한 경우)
        self.script_index = self._build_script_index_from_json(self.json_file)

    def _build_script_index_from_json(self, json_file: str) -> dict:
        """
        JSON 파일 내의 script 정보를 읽어, video_name과 start 기준으로 정렬된 index를 구성합니다.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scripts = data.get("scripts", [])
        script_index = {}
        for script in scripts:
            video_name = script.get("video_name")
            if video_name and video_name.endswith(".mp4"):
                video_name = video_name[:-4]
            if not video_name:
                continue
            try:
                start_time = float(script["start"])
                end_time = float(script["end"])
            except Exception as e:
                print(f"script 파싱 오류: {e}")
                continue

            # 고유 식별자로 video_name과 start_time 결합
            identifier = f"{video_name}_{start_time}"
            script_entry = {
                "identifier": identifier,
                "video_name": video_name,
                "start": start_time,
                "end": end_time,
                "summary": script.get("summary"),
                "score": 0.0,
            }
            if video_name not in script_index:
                script_index[video_name] = {"start": [], "scripts": []}
            script_index[video_name]["start"].append(start_time)
            script_index[video_name]["scripts"].append(script_entry)
        for vid in script_index:
            combined = list(
                zip(script_index[vid]["start"], script_index[vid]["scripts"])
            )
            combined.sort(key=lambda x: x[0])
            script_index[vid]["start"] = [item[0] for item in combined]
            script_index[vid]["scripts"] = [item[1] for item in combined]
        return script_index

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        사용자 쿼리에 대해 script 임베딩 유사도를 계산하여 상위 top_k 결과를 반환합니다.
        고유 식별자는 video_name과 start를 결합한 문자열입니다.
        """
        query_vec = self._encode_texts([user_query], disable=True)
        scores = self._compute_similarity(query_vec, self.embeddings)
        scores_np = scores.cpu().numpy()
        # 각 script 항목의 고유 식별자로 score 매핑 생성 (여기서는 start 필드를 사용)
        score_mapping = {
            f"{self.data_info[i]['video_name']}_{self.data_info[i]['start']}": float(
                score
            )
            for i, score in enumerate(scores_np)
        }
        top_idxs = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            info = self.data_info[idx]
            identifier = f"{info['video_name']}_{info['start']}"
            results.append(
                {
                    "rank": rank,
                    "video_id": info.get("video_name"),
                    "script_start_time": info["start"],
                    "script_end_time": info["end"],
                    "script_summary": info["summary"],
                    "script_id": identifier,
                    "score": score_mapping[identifier],
                }
            )
        return results


# ##############################################
# # SCRIPTSPARSERetrieval: JSON의 script 정보("scripts" 리스트, 각 항목의 요약은 "summary") 기반 sparse retrieval
# ##############################################
# nlp = spacy.load("en_core_web_sm")

# class SCRIPTSPARSERetrieval:
#     def __init__(self, config_path: str = "config/merged_config.yaml"):
#         # 설정 파일 로드
#         self.config = self._load_config(config_path)["bge-script"]
#         self.json_file = self.config["json_file"]

#         # JSON 파일에서 script 데이터 로드
#         with open(self.json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.data_info = data.get("scripts", [])

#         # script 전용 인덱스 구축 (dense에서 사용하던 구조 그대로 유지)
#         self.script_index = self._build_script_index_from_json(self.json_file)

#         # script summary에서 명사만 추출한 후 벡터화
#         self.texts = [self._extract_nouns(item["summary"]) for item in self.data_info]
#         self.vectorizer = TfidfVectorizer(stop_words="english")
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

#     def _load_config(self, config_path: str) -> dict:
#         """설정 파일 로드"""
#         with open(config_path, "r", encoding="utf-8") as f:
#             return yaml.safe_load(f)

#     def _extract_nouns(self, text: str) -> str:
#         """
#         주어진 텍스트에서 명사(NOUN, PROPN)만 추출하여 반환
#         """
#         doc = nlp(text)
#         nouns = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
#         return " ".join(nouns)

#     def _build_script_index_from_json(self, json_file: str) -> dict:
#         """
#         JSON 파일 내의 script 정보를 읽어, video_name과 start 기준으로 정렬된 index를 구축
#         (기존 dense 방식의 `_build_script_index_from_json` 유지)
#         """
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         scripts = data.get("scripts", [])
#         script_index = {}
#         for script in scripts:
#             video_name = script.get("video_name")
#             if video_name and video_name.endswith(".mp4"):
#                 video_name = video_name[:-4]
#             if not video_name:
#                 continue
#             try:
#                 start_time = float(script["start"])
#                 end_time = float(script["end"])
#             except Exception as e:
#                 print(f"script 파싱 오류: {e}")
#                 continue

#             # 고유 식별자로 video_name과 start_time 결합
#             identifier = f"{video_name}_{start_time}"
#             script_entry = {
#                 "identifier": identifier,
#                 "video_name": video_name,
#                 "start": start_time,
#                 "end": end_time,
#                 "summary": script.get("summary"),
#                 "score": 0.0,
#             }
#             if video_name not in script_index:
#                 script_index[video_name] = {"start": [], "scripts": []}
#             script_index[video_name]["start"].append(start_time)
#             script_index[video_name]["scripts"].append(script_entry)

#         # start_time 기준 정렬
#         for vid in script_index:
#             combined = list(zip(script_index[vid]["start"], script_index[vid]["scripts"]))
#             combined.sort(key=lambda x: x[0])
#             script_index[vid]["start"] = [item[0] for item in combined]
#             script_index[vid]["scripts"] = [item[1] for item in combined]

#         return script_index

#     def retrieve(self, user_query: str, top_k: int = 5) -> list:
#         """
#         사용자 쿼리에서 명사만 추출한 후 TF-IDF 벡터화하여 sparse retrieval 수행
#         (기존 dense 방식 제거 후 sparse 방식 적용)
#         """
#         query_nouns = self._extract_nouns(user_query)
#         query_vec = self.vectorizer.transform([query_nouns])
#         scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
#         top_idxs = scores.argsort()[-top_k:][::-1]

#         # 각 script 항목의 고유 식별자로 score 매핑
#         score_mapping = {
#             f"{self.data_info[i]['video_name']}_{self.data_info[i]['start']}": float(scores[i])
#             for i in range(len(scores))
#         }

#         results = []
#         for rank, idx in enumerate(top_idxs, start=1):
#             info = self.data_info[idx]
#             identifier = f"{info['video_name']}_{info['start']}"
#             results.append(
#                 {
#                     "rank": rank,
#                     "video_id": info.get("video_name"),
#                     "script_start_time": info["start"],
#                     "script_end_time": info["end"],
#                     "script_summary": info["summary"],
#                     "script_id": identifier,
#                     "score": score_mapping[identifier],
#                 }
#             )
#         return results
    

##############################################
# ImageRetrieval: 이미지 임베딩 관련 공통 기능
##############################################
class ImageRetrieval:
    def __init__(self, config_path: str, config_section: str) -> None:
        full_config = self._load_config(config_path)
        if config_section not in full_config:
            raise KeyError(
                f"'{config_section}' 설정이 {config_path}에 존재하지 않습니다."
            )
        self.config = full_config[config_section]
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.image_folder = self.config["image_folder"]
        self.embedding_file = self.config["embedding_file"]
        self.image_extensions = tuple(self.config["image_extensions"])
        self.image_filenames = []
        self.image_embeddings = None
        self._ensure_output_dir()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _ensure_output_dir(self) -> None:
        out_dir = os.path.dirname(self.embedding_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def collect_image_filenames(self) -> None:
        self.image_filenames = [
            os.path.join(self.image_folder, fname)
            for fname in sorted(os.listdir(self.image_folder))
            if fname.endswith(self.image_extensions)
        ]


##############################################
# CLIPRetrieval: CLIP을 사용한 이미지 임베딩 및 검색
##############################################
class CLIPRetrieval(ImageRetrieval):
    def __init__(self, config_path: str = "config/image_retrieval_config.yaml") -> None:
        super().__init__(config_path, config_section="clip")
        self.model, self.preprocess = clip.load(
            self.config["model_name"], device=self.device
        )
        self.model.eval()
        if os.path.exists(self.embedding_file):
            self._load_image_embeddings()
        else:
            self._extract_and_save_image_embeddings()

    def _extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(self.image_filenames, self.preprocess)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeds_list, files_list = [], []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting CLIP embeddings"):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                embeds_list.append(features.cpu())
                files_list.extend(paths)
        self.image_embeddings = torch.cat(embeds_list, dim=0)
        torch.save(
            {"filenames": files_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"CLIP 임베딩을 {self.embedding_file}에 저장했습니다.")

    def _load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"CLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embedding.T).squeeze(
            1
        )
        top_idxs = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_idxs, start=1)
        ]


##############################################
# BLIPRetrieval: BLIP2를 사용한 이미지 임베딩 및 검색
##############################################
class BLIPRetrieval(ImageRetrieval):
    def __init__(self, config_path: str = "config/image_retrieval_config.yaml") -> None:
        super().__init__(config_path, config_section="blip")
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type=self.config.get("model_type", "pretrain"),
                is_eval=True,
                device=self.device,
            )
        )
        self.model.eval()
        if os.path.exists(self.embedding_file):
            self._load_image_embeddings()
        else:
            self._extract_and_save_image_embeddings()

    def _extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(
            self.image_filenames, self.vis_processors["eval"], convert_mode="RGB"
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeds_list, files_list = [], []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting BLIP embeddings"):
                images = images.to(self.device)
                features = self.model.extract_features({"image": images}, mode="image")
                image_embed = features.image_embeds_proj.mean(dim=1)
                image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
                embeds_list.append(image_embed.cpu())
                files_list.extend(paths)
        self.image_embeddings = torch.cat(embeds_list, dim=0)
        torch.save(
            {"filenames": files_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"BLIP 임베딩을 {self.embedding_file}에 저장했습니다.")

    def _load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"BLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        text_input = self.txt_processors["eval"](query)
        sample = {"text_input": [text_input]}
        with torch.no_grad():
            features_text = self.model.extract_features(sample, mode="text")
            text_embed = features_text.text_embeds_proj
            if text_embed.dim() == 3 and text_embed.size(1) > 1:
                text_embed = text_embed.mean(dim=1)
            elif text_embed.dim() != 2:
                raise ValueError(
                    f"예상하지 못한 텍스트 임베딩 차원: {text_embed.shape}"
                )
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embed.T).squeeze(1)
        top_idxs = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_idxs, start=1)
        ]


##############################################
# Rankfusion: CLIP, BLIP, Scene retrieval 점수를 융합하여 최종 결과 생성
##############################################
class Rankfusion:
    """
    Rankfusion은
      1) CLIP,
      2) BLIP,
      3) Scene retrieval (BGE),
      4) Script retrieval (BGE)
    점수를 융합하여 최종 결과를 생성합니다.

    (1) 이미지 파일명 예시: "video1_12.3.jpg"  -> video_id="video1", timestamp=12.3
    (2) scene_retriever.scene_index[video_id]에서 (start_time, end_time)에 걸쳐 있는 scene을 찾아 scene_score를 가져옵니다.
    (3) script_retriever.script_index[video_id]에서 (start, end)에 걸쳐 있는 script를 찾아 script_score를 가져옵니다.
    (4) 최종 score = weight_clip*clip_score + weight_blip*blip_score
                    + weight_scene*scene_score + weight_script*script_score
    """

    def __init__(
        self,
        scene_retriever,
        script_retriever,
        clip_retriever,
        blip_retriever,
        weight_clip: float = 0.25,
        weight_blip: float = 0.25,
        weight_scene: float = 0.25,
        weight_script: float = 0.25,
    ):
        self.scene_retriever = scene_retriever
        self.script_retriever = script_retriever
        self.clip_retriever = clip_retriever
        self.blip_retriever = blip_retriever

        self.weight_clip = weight_clip
        self.weight_blip = weight_blip
        self.weight_scene = weight_scene
        self.weight_script = weight_script

        # CLIP 임베딩을 빠르게 참조하기 위해 미리 딕셔너리 형태로 캐싱
        self._clip_embedding_dict = None

    def _get_clip_embedding_dict(self) -> dict:
        """
        CLIP 임베딩을 {파일명: numpy 배열} 형태로 만들어 캐싱합니다.
        (Diverse selection 시 clustering에 사용)
        """
        if self._clip_embedding_dict is None:
            self._clip_embedding_dict = {
                fname: emb.cpu().numpy()
                for fname, emb in zip(
                    self.clip_retriever.image_filenames,
                    self.clip_retriever.image_embeddings,
                )
            }
        return self._clip_embedding_dict

    def _find_scene_for_image(self, video_id: str, timestamp: float) -> dict:
        """
        주어진 video_id, timestamp로 scene_index에서 해당 scene을 찾아 반환합니다.
        (이진탐색 bisect 사용)
        """
        scene_index = self.scene_retriever.scene_index
        if video_id not in scene_index:
            return None
        starts = scene_index[video_id]["starts"]
        scenes = scene_index[video_id]["scenes"]
        idx = bisect.bisect_right(starts, timestamp) - 1
        if idx >= 0:
            scene = scenes[idx]
            if timestamp <= scene["end_time"]:
                return scene
        return None

    def _find_script_for_image(self, video_id: str, timestamp: float) -> dict:
        """
        주어진 video_id, timestamp로 script_index에서 해당 script 구간을 찾아 반환합니다.
        (이진탐색 bisect 사용)
        """
        script_index = self.script_retriever.script_index
        if video_id not in script_index:
            return None
        starts = script_index[video_id]["start"]
        scripts = script_index[video_id]["scripts"]
        idx = bisect.bisect_right(starts, timestamp) - 1
        if idx >= 0:
            s = scripts[idx]
            if timestamp <= s["end"]:
                return s
        return None

    def _get_scene_scores(
        self, image_filenames: set, scene_score_mapping: dict
    ) -> dict:
        """
        이미지 파일명 -> scene score 매핑을 구해서 반환합니다.
        (예: "video1_12.3.jpg" -> scene_index 통해 매칭되는 scene_id -> scene_score)
        """
        scores = {}
        for img in image_filenames:
            try:
                basename = os.path.basename(img)
                file_stem, _ = os.path.splitext(basename)
                video_id, ts_str = file_stem.rsplit("_", 1)
                timestamp = float(ts_str)
            except Exception as e:
                print(f"[Scene] 파일명 파싱 오류 ({img}): {e}")
                scores[img] = 0.0
                continue

            scene = self._find_scene_for_image(video_id, timestamp)
            if scene:
                scene_id = scene.get("scene_id")
                scores[img] = scene_score_mapping.get(scene_id, 0.0)
            else:
                scores[img] = 0.0

        return scores

    def _get_script_scores(
        self, image_filenames: set, script_score_mapping: dict
    ) -> dict:
        """
        이미지 파일명 -> script score 매핑을 구해서 반환합니다.
        (예: "video_name_74.36.jpg" -> script_index 통해 매칭되는 구간 -> script_score)
        """
        scores = {}
        for img in image_filenames:
            try:
                basename = os.path.basename(img)
                file_stem, _ = os.path.splitext(basename)
                video_id, ts_str = file_stem.rsplit("_", 1)
                timestamp = float(ts_str)
            except Exception as e:
                print(f"[Script] 파일명 파싱 오류 ({img}): {e}")
                scores[img] = 0.0
                continue

            script = self._find_script_for_image(video_id, timestamp)
            if script:
                # 예: identifier = f"{video_name}_{start}"
                identifier = script.get("identifier")
                scores[img] = script_score_mapping.get(identifier, 0.0)
            else:
                scores[img] = 0.0

        return scores

    def retrieve(self, query: str, top_k: int = 10, union_top_n: int = None) -> list:
        if union_top_n is None:
            union_top_n = max(
                len(self.clip_retriever.image_filenames),
                len(self.blip_retriever.image_filenames),
            )
        # 1) CLIP
        clip_results = self.clip_retriever.retrieve(query, top_k=union_top_n)
        clip_dict = {res["image_filename"]: res["score"] for res in clip_results}

        # 2) BLIP
        blip_results = self.blip_retriever.retrieve(query, top_k=union_top_n)
        blip_dict = {res["image_filename"]: res["score"] for res in blip_results}

        # 3) Scene retrieval
        scene_results = self.scene_retriever.retrieve(query, top_k=len(self.scene_retriever.data_info))
        scene_score_mapping = {res["scene_id"]: res["score"] for res in scene_results}

        # 4) Script retrieval
        script_results = self.script_retriever.retrieve(query, top_k=len(self.script_retriever.data_info))
        script_score_mapping = {res["script_id"]: res["score"] for res in script_results}

        # 이미지 전체 집합
        all_imgs = set(clip_dict.keys()).union(set(blip_dict.keys()))
        scene_dict = self._get_scene_scores(all_imgs, scene_score_mapping)
        script_dict = self._get_script_scores(all_imgs, script_score_mapping)

        # 최종 Fusion 결과 생성 (여기서 각 이미지에 대해 scene 정보를 추가)
        fused = []
        for img in all_imgs:
            s_clip = clip_dict.get(img, 0.0)
            s_blip = blip_dict.get(img, 0.0)
            s_scene = scene_dict.get(img, 0.0)
            s_script = script_dict.get(img, 0.0)

            fusion_score = (
                self.weight_clip * s_clip
                + self.weight_blip * s_blip
                + self.weight_scene * s_scene
                + self.weight_script * s_script
            )
            
            # 이미지 파일명에서 video_id와 timestamp 추출 (예: "video1_12.3.jpg")
            try:
                basename = os.path.basename(img)
                file_stem, _ = os.path.splitext(basename)
                video_id, ts_str = file_stem.rsplit("_", 1)
                timestamp = float(ts_str)
            except Exception as e:
                print(f"[Rankfusion] 파일명 파싱 오류 ({img}): {e}")
                video_id, timestamp = None, None

            # 해당 이미지의 scene 정보 조회 (없으면 None 반환)
            scene_info = None
            if video_id is not None and timestamp is not None:
                scene_info = self._find_scene_for_image(video_id, timestamp)

            fused.append(
                {
                    "image_filename": img,
                    "clip_score": s_clip,
                    "blip_score": s_blip,
                    "scene_score": s_scene,
                    "script_score": s_script,
                    "fusion_score": fusion_score,
                    "scene_info": scene_info  # scene_info에 scene 상세 정보를 포함
                }
            )

        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        for i, item in enumerate(fused, start=1):
            item["rank"] = i
        return fused[:top_k]


    def select_diverse_results_by_clustering(
        self,
        retrieval_results: list[dict],
        desired_num: int = 5,
        top_n: int = 100,
    ) -> list[dict]:
        """
        retrieval_results: retrieval 시스템이 반환한 결과 리스트 (예: fusion_score 기준 내림차순 정렬)
        desired_num: 최종 선택할 결과 수 (예: 5)
        top_n: diversity 적용을 위한 후보 수 (예: 상위 100개 결과)
        """
        # 후보 집합 (상위 top_n 결과)
        candidates = retrieval_results[:top_n]

        # 내부적으로 CLIP 임베딩 dict 생성
        embedding_dict = self._get_clip_embedding_dict()

        features = []
        candidate_filenames = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in embedding_dict:
                features.append(embedding_dict[fname])
                candidate_filenames.append(fname)
        if len(features) == 0:
            return []

        features = np.stack(features, axis=0)

        # 클러스터 수를 desired_num으로 설정하여 KMeans 클러스터링 수행
        kmeans = KMeans(n_clusters=desired_num, random_state=0).fit(features)
        labels = kmeans.labels_

        diverse_results = []
        # 각 클러스터별로 최고 fusion_score를 가진 후보 선택
        for cluster_id in range(desired_num):
            cluster_candidates = [
                candidates[i] for i, label in enumerate(labels) if label == cluster_id
            ]
            if not cluster_candidates:
                continue
            best_candidate = max(cluster_candidates, key=lambda x: x["fusion_score"])
            diverse_results.append(best_candidate)

        diverse_results.sort(key=lambda x: x["fusion_score"], reverse=True)

        return diverse_results

    def select_diverse_results(
        self,
        retrieval_results: list[dict],
        desired_num: int = 10,
        similarity_threshold: float = 0.9,
    ) -> list[dict]:
        """
        retrieval_results: retrieval 시스템이 반환한 결과 리스트 (예: fusion_score 기준 내림차순 정렬)
        desired_num: 최종 선택할 결과 수 (예: 10)
        similarity_threshold: 두 이미지 간 코사인 유사도가 이 값보다 높으면 너무 유사하다고 판단 (0~1 사이 값)

        CLIP 임베딩(이미 정규화된 벡터)을 사용하여, 이미 선택된 이미지와의 코사인 유사도가 similarity_threshold를 초과하면 후보에서 제외합니다.
        """
        # CLIP 임베딩 dict 생성 (이미 캐싱되어 있다면 캐싱값 사용)
        embedding_dict = self._get_clip_embedding_dict()

        # fusion_score 기준 내림차순 정렬 (만약 retrieval_results가 이미 정렬되어 있다면 이 단계는 생략 가능)
        sorted_results = sorted(
            retrieval_results, key=lambda x: x["fusion_score"], reverse=True
        )

        selected_results = []
        for res in sorted_results:
            candidate_fname = res["image_filename"]
            # 후보 이미지의 임베딩
            candidate_emb = embedding_dict.get(candidate_fname)
            if candidate_emb is None:
                continue

            too_similar = False
            for sel in selected_results:
                sel_emb = embedding_dict.get(sel["image_filename"])
                # 이미 정규화된 벡터이므로 내적이 코사인 유사도
                similarity = np.dot(candidate_emb, sel_emb)
                if similarity > similarity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected_results.append(res)
            if len(selected_results) >= desired_num:
                break

        return selected_results


##############################################
# 메인 실행 (Usage Example)
##############################################
if __name__ == "__main__":
    text_query = "A woman grabs a man from behind and holds a knife to his throat, threatening him."
    config_path = "config/video_retrieval_config.yaml"  # video 및 image retrieval 설정 파일 (예: merged_config.yaml)

    # # (1) 초기화: Scene, Script, CLIP, BLIP
    scene_retriever = SCENERetrieval(config_path=config_path)
    script_retriever = SCRIPTRetrieval(config_path=config_path)
    clip_retriever = CLIPRetrieval(config_path=config_path)
    blip_retriever = BLIPRetrieval(config_path=config_path)

    # # top_k 개수만큼 검색 결과를 가져옴 (예제에서는 10개)
    # clip_results = clip_retriever.retrieve(query=text_query, top_k=10000)
    
    # # 결과 출력: 각 결과에 대해 rank, image_filename, score 출력
    # print("=== CLIP Retrieval Results ===")
    # for res in clip_results:
    #     print(f"Rank {res['rank']}: {res['image_filename']} - score: {res['score']:.4f}")

    # (2) Rankfusion 초기화 (script_retriever 추가)
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        script_retriever=script_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=3,
        weight_blip=3,
        weight_scene=1,
        weight_script=0,
    )

    # (3) Fusion 검색
    fusion_results = rankfusion.retrieve(query=text_query, top_k=1000, union_top_n=100000)
    print("=== Fusion 검색 결과 (요약) ===")
    for res in fusion_results[:30]:
        # scene_info가 결과에 포함되어 있다면 scene_id 추출, 없으면 None 처리
        scene_id = res.get("scene_info", {}).get("scene_id") if res.get("scene_info") else None
        print(
            f"Rank {res['rank']}: {res['image_filename']} | "
            f"clip={res['clip_score']:.4f}, blip={res['blip_score']:.4f}, "
            f"scene={res['scene_score']:.4f}, script={res['script_score']:.4f}, "
            f"fusion={res['fusion_score']:.4f} | scene_id={scene_id}"
        )

    # (4) Diverse 결과 예시
    diverse_results = rankfusion.select_diverse_results_by_clustering(
        fusion_results, desired_num=10, top_n=100
    )
    print("\n=== 클러스터링 기반 Diversity 적용 후 최종 결과 ===")
    for res in diverse_results:
        # scene_info가 결과에 포함되어 있다면 scene_id 추출, 없으면 None 처리
        scene_id = res.get("scene_info", {}).get("scene_id") if res.get("scene_info") else None
        print(
            f"Rank {res['rank']}: {res['image_filename']} | "
            f"clip={res['clip_score']:.4f}, blip={res['blip_score']:.4f}, "
            f"scene={res['scene_score']:.4f}, script={res['script_score']:.4f}, "
            f"fusion={res['fusion_score']:.4f} | scene_id={scene_id}"
        )
