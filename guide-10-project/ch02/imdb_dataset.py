from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, pipeline
from transformers import AutoTokenizer

# IMDb 데이터셋 불러오기
print("데이터셋 받기...")
imdb_dataset = load_dataset("stanfordnlp/imdb")

# 사전 학습 모델과 토크나이저 불러오기 - 토크나이저와 모델은 쌍으로 사용하니 이름이 같다...
print("토크나이저와 모델 설정...")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# 이렇게 Auto 붙은 걸 쓰면 적당한 변형을 찾아준다고...
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("감정분석 파이프라인 설정")
sentiment_analysis_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer
)

# 데이터셋에서 샘플 텍스트 추출
sample_text = imdb_dataset["test"][0]["text"]

# 샘플 텍스트에 감성 분석 적용
result = sentiment_analysis_pipeline(sample_text)

# 결과 출력 - 아무 학습도 안했지만 미리 학습된 모델이라 엄청 자신있게 맞춘다...
print("Sample Text:", sample_text)
print("Sentiment Analysis Result:", result)
