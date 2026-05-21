from transformers import AutoTokenizer, AutoModelForTokenClassification

# 토큰 단위로 분류한다...예를 들면 이게 무슨 이름인지...또는 품사가 뭔지...뭐 이런거?
# 일단 마찬가지로 BERT 모델 만들고...
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 이것도 num_labels 없다...기본적으로 다중입력, 다중분류는 그렇다는데...
model = AutoModelForTokenClassification.from_pretrained(model_name)
print(model)

from datasets import load_dataset

# KLUE에서 개체명 인식 데이터셋을 받는다는데...
dataset = load_dataset("klue", "ner")

# 근데 이건 뭐 그냥 글자 하나씩 끊었는데...거기 뭔 개체명이 있다고 태그를 붙여놨다네...
sample = dataset["train"][0]
print("tokens : ", sample["tokens"][:20])
print("ner tags : ", sample["ner_tags"][:20])
print((len(sample["tokens"]), len(sample["tokens"])))

for i in range(len(sample["ner_tags"])):
    print(sample["tokens"][i], "\t", sample["ner_tags"][i])


# 뭘 하는지 잘 모르겠고, 그냥 늘 토큰나이저 만들고 map한다...
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 토큰을 해당 단어에 매핑
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # 스페셜 토큰을 -100으로 세팅
            if word_idx is None:
                label_ids.append(12)
                # label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # 주어진 단어의 첫 번째 토큰에만 레이블을 지정
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 길이 맞추는 것도 늘 하는건데, 토큰 분류에서는 특수 토큰이 -100이 되야 하니까...ForToken 분류 모델을 쓴다..
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_dataset["train"][i] for i in range(10)])

from transformers import AutoModelForTokenClassification

# klue/bert 모델은 id2label이 없단다...그래서 설정해서 넣어준단다...
# 앞쪽 알파벳으로, B는 시작, I는 내부, O는 모르겠는 것들...
# DT-날짜, LC-위치, 0G-단체, PS-사람, QT-수량, Tl-시간
# 개체명
id2label = {
    0: "B-DT",
    1: "I-DT",
    2: "B-LC",
    3: "I-LC",
    4: "B-OG",
    5: "I-OG",
    6: "B-PS",
    7: "I-PS",
    8: "B-QT",
    9: "I-QT",
    10: "B-TI",
    11: "I-TI",
    12: "O",
}
label2id = {
    "B-DT": 0,
    "I-DT": 1,
    "B-LC": 2,
    "I-LC": 3,
    "B-OG": 4,
    "I-OG": 5,
    "B-PS": 6,
    "I-PS": 7,
    "B-QT": 8,
    "I-QT": 9,
    "B-TI": 10,
    "I-TI": 11,
    "O": 12,
}
model = AutoModelForTokenClassification.from_pretrained(
    "klue/bert-base", num_labels=13, id2label=id2label, label2id=label2id
)

import torch

# 이것도 뭐 학습도 미세 조정도 없고 그냥 예측한다..
with torch.no_grad():
    logits = model(**batch).logits

predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
print(predicted_token_class)

import evaluate

# 평가해보면...0.02...그냥 눈 감고 찍으니만 못하다...이런걸 왜 하지?
pred_labels = logits.argmax(dim=-1).view(-1).cpu().numpy()
true_labels = batch["labels"].view(-1).numpy()
pred_labels.shape, true_labels.shape

f1 = evaluate.load("f1")
print(f1.compute(predictions=pred_labels, references=true_labels, average="micro"))
