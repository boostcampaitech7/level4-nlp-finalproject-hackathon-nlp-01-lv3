# 벤치마크 평가

## 소개
이 저장소는 FAISS 기반의 유사도 검색과 BAAI/bge-m3 모델 임베딩을 사용하여 비디오 장면 검색을 평가하는 스크립트를 제공합니다. 데이터 병합, 임베딩 생성, 검색, 평가의 단계로 구성됩니다.

## 설치
스크립트를 실행하기 전에 필요한 종속성을 설치하세요:

```sh
pip install -r requirements.txt
```

## 설정
파일 경로 및 실행 단계를 설정하려면 `config.yaml`을 수정하세요.

```yaml
paths:
  frame_json_path: "./json/frame_output_v0.json"
  scene_json_path: "./json/scene_output_v0.json"
  merged_json_path: "./json/merged_output_v0.json"
  embedding_json_path: "./embedding/emb_v0.json"
  input_csv_path: "./test_dataset/own_dataset_v2.csv"
  result_csv_path: "./result/result_v0.csv"
  output_score_csv_path: "./result/eval_v0.csv"

settings:
  model_name: "BAAI/bge-m3"
  top_k: 1
  start_stage: "merge"  # [merge, embedding, retrieving, evaluation]
```

## 파이프라인 실행
스크립트는 `config.yaml`의 `start_stage` 설정에 따라 다양한 처리 단계를 지원합니다.

### 1. 프레임 및 장면 데이터 병합
```sh
python benchmark_evaluation.py
```
이 단계에서는 프레임과 장면 JSON 데이터를 하나의 형식으로 병합합니다.

### 2. 임베딩 생성
`config.yaml`에서 `start_stage`를 `embedding`으로 변경한 후 실행하세요:
```sh
python benchmark_evaluation.py
```
이 단계에서는 병합된 데이터셋에 대한 임베딩을 생성합니다.

### 3. 유사한 장면 검색
`config.yaml`에서 `start_stage`를 `retrieving`으로 변경한 후 실행하세요:
```sh
python benchmark_evaluation.py
```
이 단계에서는 FAISS 유사도 검색을 기반으로 가장 관련성이 높은 장면을 검색합니다.

### 4. 결과 평가
`config.yaml`에서 `start_stage`를 `evaluation`으로 변경한 후 실행하세요:
```sh
python benchmark_evaluation.py
```
이 단계에서는 검색된 결과를 정답 데이터와 비교하여 성능을 평가합니다.

## 평가 모드
기본적으로 `top_k` 개수만큼 평가가 수행됩니다. 오직 `top-1` 결과만 평가하려면 `config.yaml`을 다음과 같이 수정하세요:
```yaml
settings:
  top_k: 1
```
이렇게 하면 각 쿼리에 대해 첫 번째 검색 결과만 고려하여 평가가 진행됩니다.

## 출력 파일
- `merged_json_path`: 병합된 프레임 및 장면 데이터 저장
- `embedding_json_path`: 생성된 텍스트 임베딩 저장
- `result_csv_path`: 검색된 결과 저장
- `output_score_csv_path`: 평가 점수 저장

## 참고 사항
- 실행 전에 `config.yaml`이 올바르게 설정되었는지 확인하세요.
- `start_stage` 설정에 따라 필요하지 않은 단계는 자동으로 건너뜁니다.
- FAISS는 효율적인 최근접 이웃 검색을 위해 사용됩니다.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

