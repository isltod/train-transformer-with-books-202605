# 이런 식으로 어떤 측정함수든 만들어 사용할 수 있고...
import os


def simple_accuracy(preds, labels):
    return {"accuracy": (preds == labels).to(float).mean().item()}


import evaluate


# 이런 식으로 만들어서 학습에 적용하는데...사실 이건 그냥 원래 걸 써도 될거 같은데...
def custom_metrics(pred):
    f1 = evaluate.load("f1")
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return f1.compute(predictions=preds, references=labels, average="micro")


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator
)

# 늘 하듯, 모델 만들고
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

# 데이터셋 받고
dataset = load_dataset("klue", "ynat")


def tokenize_function(sample):
    result = tokenizer(
        sample["title"],
        padding="max_length",
    )
    return result


# dataset.map으로 토큰화
datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["guid", "title", "url", "date"]
)
print(datasets)

# TrainingArguments 설정하는데, 이건 앞에 코드 보고 수정 좀 해야 하고...
path = "e:/Devs/train-transformer-with-books-202605/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU만 사용
os.environ["TENSORBOARD_LOGGING_DIR"] = path + "data/logs"
args = TrainingArguments(
    # 계속 죽다가 여기부터 아래까지 바꾸니 돌아 간다...8도 된다...결국 gra.. 아니면 fp 설정이 도움이 된다는 얘기...
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # 배치 사이즈 대신 누적 스텝 활용
    gradient_checkpointing=True,  # Gradient Checkpointing 활성화
    fp16=True,  # 혼합 정밀도 학습 (메모리 절약)
    # 여기까지....
    learning_rate=5e-5,
    max_steps=500,
    # 이건 아래처럼 바뀌고
    # evaluation_strategy="steps",
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=50,
    # 로그 경로 설정은 위에 환경변수 설정으로 변경...
    # logging_dir="/content/logs",
    save_strategy="steps",
    save_steps=50,
    output_dir=path + "data/ckpt",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    # 이것도 아래처럼 바뀌고
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=custom_metrics,
)

trainer.train()
