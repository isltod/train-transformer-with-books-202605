from pprint import pprint

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 사전 학습된 klue/bert-base 모델로 문장 분류 시도
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 사전 학습과 다른 설정은 여기서 맞추는데, 예를 들어 num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# 모델은 큼지막하게, BertEmbeddings, BertEncoder, BertPooler, Dropout, classifier...
pprint(model)
# 학습 안된 상태에서도 라벨은 설정대로 2개
pprint(model.config.id2label)
# 별 의미없이 문장 분류
inputs = tokenizer("안녕? 내 강아지는 귀여워.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])

# KLUE 문장 유사도 STS 데이터셋...뭔지...
from datasets import load_dataset

dataset = load_dataset("klue", "sts")
pprint(dataset["train"])


# 늘 하듯,dataset.map으로 토크나이저 연결해서 데이터셋 만들고...
def process_data(batch):
    # 문장1, 문장2 쌍으로 넣고
    result = tokenizer(batch["sentence1"], text_pair=batch["sentence2"])
    # 정답지
    result["labels"] = [x["binary-label"] for x in batch["labels"]]
    return result


dataset = dataset.map(
    process_data,
    batched=True,
    # 원래 있던 컬럼은 다 제거...
    remove_columns=dataset["train"].column_names,
)

# 학습하려면 문장 길이를 맞춰야 한다고...
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer)

# 테스트 10개 추론...학습도 안하고...
batch = collator([dataset["train"][i] for i in range(10)])
with torch.no_grad():
    # 이게 예측이겠지...
    logits = model(**batch).logits
pprint(logits)

# 평가를 한다는데...아직 학습도 안했는데...했다 치는건가...
# 위에서 예측 결과와 정답 10개
pred_labels = logits.argmax(dim=1).cpu().numpy()
true_labels = batch["labels"].numpy()
print(pred_labels)
print(true_labels)

import evaluate

# f1 스코어로 평가 - 역시나 때려잡기로 반반 맞춘다...
f1 = evaluate.load("f1")
pprint(f1.compute(predictions=pred_labels, references=true_labels, average="micro"))

# 이건 논외로 갑자기 회귀 모델을 만들어보겠다며...생뚱맞기가 그지없는데...
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
# 아무튼 핵심은 num_labels 값으로,
# 이걸 2 이상으로 주면 알아서 분류 모델로 변신해서 출력은 소프트맥스로, 손실은 크로스 엔트로피로...
# 이걸 1로 주면 알아서 회귀로 변해서, 출력은 소프트맥스 들어가기 전 로짓값으로, 손실은 MSE로 처리한다고...
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=1)
print(model)

with torch.no_grad():
    logits = model(**batch).logits
pprint(logits)
