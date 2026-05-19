# 파이프라인
from transformers import pipeline

# 그냥 파이프라인 받아서 하기...
pipe = pipeline(task="text-classification", model="google-bert/bert-base-uncased")
print(pipe("유튜브 내달 2일까지 크리에이터 지원 공간 운영"))

# 미세조정 모델 경로로 가져오기
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 저장한 모델과 토크나이저로 파이프라인 만들기...
# 이렇게 경로명을 모델명으로 넘기면 그 경로에서 찾아오나....
model_name = "data/MyBertModel/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pipe = pipeline(task="text-classification", tokenizer=tokenizer, model=model)
print(pipe("유튜브 내달 2일까지 크리에이터 지원 공간 운영"))

# 오트클래스로 파이프라인 만들기...
model_name = "google-bert/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
print(pipeline("유튜브 내달 2일까지 크리에이터 지원 공간 운영"))

# 파이프라인을 직접 구현하기 - 이게 코드가 길어진다고...
import torch

model_name = "google-bert/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.cuda().eval()

with torch.no_grad():
    output = model(
        **tokenizer(
            "유튜브 내달 2일까지 크리에이터 지원 공간 운영", return_tensors="pt"
        ).to(model.device)
    )
    result = torch.softmax(output.logits.cpu(), -1)

result = [
    {"label": f"LABEL_{l}", "score": result[i, l].item()}
    for i, l in enumerate(result.argmax(-1))
]
print(result)
