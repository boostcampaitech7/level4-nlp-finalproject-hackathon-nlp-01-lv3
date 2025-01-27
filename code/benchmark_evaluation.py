import json
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# 설정
frame_json_path = "./json/frame_output_v3.json"                # frame version
clip_json_path = "./json/clip_output_v10.1.json"                  # clip version
merged_json_path = "./json/merged_output_v5.json"              # frame+clip merge 저장 위치
embedding_json_path = "./embedding/emb_v5.json"                # embedding 저장 위치
input_csv_path = "./test_dataset/own_dataset_v2.csv"           # test dataset 위치
result_csv_path = "./result/result_v5.csv"                     # retrieve result 저장 위치
output_score_csv_path = "./result/eval_v5.csv"                 # 평가용 파일 저장 위치
model_name = "BAAI/bge-m3"                                     # 임베딩 모델
top_k = 1                                                      # retrieve top k 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def timestamp_to_seconds(timestamp):
    """
    Convert a timestamp string (HH:MM:SS,sss) to seconds.
    """
    h, m, s = map(float, timestamp.replace(",", ".").split(":"))
    return round(h * 3600 + m * 60 + s, 3)

def load_json(file_path):
    """
    Load JSON data from a given file path.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    """
    Save data to a JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def merge_frame_data(frame_data):
    """
    Convert frame data to the desired format.
    """
    return [
        {
            "video_id": frame["video_id"],
            "scale": "frame",
            "start": timestamp_to_seconds(frame["timestamp"]),
            "end": timestamp_to_seconds(frame["timestamp"]),
            "description": frame["caption"],
        }
        for frame in frame_data
    ]

def merge_clip_data(clip_data):
    """
    Convert clip data to the desired format.
    """
    return [
        {
            "video_id": clip["video_id"],
            "scale": "clip",
            "start": clip["start_timestamp"],
            "end": clip["end_timestamp"],
            "description": clip["clip_description"],
        }
        for clip in clip_data
    ]

def get_embeddings(text, tokenizer, model, device):
    """
    Generate embeddings for the given text using the specified tokenizer and model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  
    return embeddings.squeeze(0).cpu().tolist()

def generate_embeddings(merged_json_path, embedding_json_path, tokenizer, model, device):
    """
    Generate embeddings for the merged data and save them to a JSON file.
    """
    data = load_json(merged_json_path)
    embedded_data = []
    for item in data:
        embedding = get_embeddings(item["description"], tokenizer, model, device)
        embedded_data.append({**item, "embedding": embedding})
    save_json(embedded_data, embedding_json_path)
    print(f"Embeddings saved to: {embedding_json_path}")

def get_query_embedding(query, tokenizer, model, device):
    """
    Generate embeddings for a query text.
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :] 
    return embedding.squeeze(0).cpu().numpy()

def search_similar_faiss(query_embedding, index, metadata, top_k=5):
    """
    Search for the most similar embeddings in the FAISS index.
    """
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [
        {"item": metadata[idx], "distance": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]

def process_queries(input_csv_path, embedding_json_path, result_csv_path, top_k, tokenizer, model, device):
    """
    Process queries and find top-k similar results using FAISS.
    """
    query_data = pd.read_csv(input_csv_path)
    embedded_data = load_json(embedding_json_path)

    embeddings = np.array([item["embedding"] for item in embedded_data]).astype("float32")
    metadata = [item for item in embedded_data]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    results = []
    for _, row in query_data.iterrows():
        query = row['query']
        query_embedding = get_query_embedding(query, tokenizer, model, device)
        top_results = search_similar_faiss(query_embedding, index, metadata, top_k)

        for result in top_results:
            results.append({
                "original_query": query,
                "retrieved_text": result["item"]["description"],
                "video_id": result["item"]["video_id"],
                "start": result["item"]["start"],
                "end": result["item"]["end"],
                "distance": result["distance"]
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {result_csv_path}")

def evaluate_results(result_csv_path, ground_truth_csv_path, output_score_csv_path, top_k=5):
    """
    Evaluate results by comparing with the ground truth.
    """
    result_data = pd.read_csv(result_csv_path)
    ground_truth_data = pd.read_csv(ground_truth_csv_path)

    scoring_results = []
    for _, row in ground_truth_data.iterrows():
        ground_truth_video_id = row["video"]
        ground_truth_start = row["start"]
        ground_truth_end = row["end"]
        index = row["index"]

        if str(ground_truth_video_id).startswith("-"):
            ground_truth_video_id = str(ground_truth_video_id)[1:]

        matching_results = result_data[result_data["original_query"] == row["query"]].head(top_k)

        is_correct = 0
        is_video_id_match = 0

        for _, match in matching_results.iterrows():
            result_video_id = match["video_id"]
            result_start = match["start"]
            result_end = match["end"]

            if str(result_video_id).startswith("-"):
                result_video_id = str(result_video_id)[1:]

            if result_video_id == ground_truth_video_id:
                is_video_id_match = 1
                if not (result_end < ground_truth_start or result_start > ground_truth_end):
                    is_correct = 1
                    break 

        scoring_results.append({
            "index": index,
            "is_correct": is_correct,
            "is_video_id_match": is_video_id_match,
        })

    scoring_results_df = pd.DataFrame(scoring_results)
    scoring_results_df.to_csv(output_score_csv_path, index=False, encoding="utf-8-sig")
    print(f"Evaluation results saved to: {output_score_csv_path}")

    ground_truth_data = ground_truth_data.drop_duplicates(subset=["index"])
    scoring_results_df = scoring_results_df.drop_duplicates(subset=["index"])
    merged_data = ground_truth_data.merge(scoring_results_df, on="index", how="inner")

    total_samples = len(merged_data)
    video_id_match_count = merged_data["is_video_id_match"].sum()
    correct_count = merged_data["is_correct"].sum()

    video_id_match_ratio = video_id_match_count / total_samples * 100
    correct_ratio = correct_count / total_samples * 100

    print("\n=== 전체 결과 ===")
    print(f"전체 데이터셋 크기: {total_samples}")
    print(f"Video ID 정답 개수: {video_id_match_count} ({video_id_match_ratio:.2f}%)")
    print(f"정답 개수: {correct_count}/{total_samples} ({correct_ratio:.2f}%)")

    type_stats = merged_data.groupby("type").agg(
        total=("type", "size"),
        video_id_match_count=("is_video_id_match", "sum"),
        correct_count=("is_correct", "sum")
    ).reset_index()

    type_stats["video_id_match_ratio"] = (type_stats["video_id_match_count"] / type_stats["total"]) * 100
    type_stats["correct_ratio"] = (type_stats["correct_count"] / type_stats["total"]) * 100

    print("\n=== Type별 결과 ===")
    for _, row in type_stats.iterrows():
        print(f"\nType: {row['type']}")
        print(f"  총 데이터: {row['total']}")
        print(f"  Video ID 정답 개수: {row['video_id_match_count']} ({row['video_id_match_ratio']:.2f}%)")
        print(f"  정답 개수: {row['correct_count']} ({row['correct_ratio']:.2f}%)")


if __name__ == "__main__":
    # topk만 바꿀 때 주석 처리 구간
    # frame_data = load_json(frame_json_path)
    # clip_data = load_json(clip_json_path)["video_clips_info"]
    # merged_data = merge_frame_data(frame_data) + merge_clip_data(clip_data)
    # save_json(merged_data, merged_json_path)
    # print(f"Merged json saved to: {merged_json_path}")

    # generate_embeddings(merged_json_path, embedding_json_path, tokenizer, model, device)
    # topk만 바꿀 때 주석 처리 구간

    process_queries(input_csv_path, embedding_json_path, result_csv_path, top_k, tokenizer, model, device)

    evaluate_results(result_csv_path, input_csv_path, output_score_csv_path, top_k=top_k)
