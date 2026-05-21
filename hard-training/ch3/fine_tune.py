import datasets
import transformers

print("transformers version:", transformers.__version__)
print("datasets version:", datasets.__version__)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "klue/bert-base"
# Auto로 시작하면 변형 중 적당한 버전을 찾아준다고..
# 근데 UNEXPECTED나 MISSING이 쫘악 뜨는 걸 보면 잘 못 찾아주는 거 같은데...
tokenizer = AutoTokenizer.from_pretrained(model_name)
# num_labels - 기사를 7개 라벨로 분류한 데이터를 사용할 모델...
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

from datasets import load_dataset

dataset = load_dataset("klue", "ynat")
# 훈련 45,678개, 시험 9,107개
print(dataset)
# 라벨 인덱스는 0~6
print(set([i for i in dataset["train"]["label"]]))


# dataset이 전처리를 할 때, 그 전처리 툴인 토큰화 부분을 이렇게 tokenizer로 만들어서 콜백으로 연결시킨다...
def tokenize_function(sample):
    result = tokenizer(
        sample["title"],
        padding="max_length",
    )
    return result


# 여기서는 map이 또 된다...
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["guid", "title", "url", "date"],
)
print(tokenized_dataset)

from transformers import Trainer, TrainingArguments, default_data_collator
import os

# train() 돌리니 죽어버리는데...이걸로 될라나...
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU만 사용

# 밑에 logging_dir 옵션을 앞으로는 이렇게...
os.environ["TENSORBOARD_LOGGING_DIR"] = "data/logs"
args = TrainingArguments(
    # 계속 죽다가 여기부터 아래까지 바꾸니 돌아 간다...8도 된다...결국 gra.. 아니면 fp 설정이 도움이 된다는 얘기...
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # 배치 사이즈 대신 누적 스텝 활용
    gradient_checkpointing=True,  # Gradient Checkpointing 활성화
    fp16=True,  # 혼합 정밀도 학습 (메모리 절약)
    # 여기까지....
    learning_rate=5e-5,
    # steps를 기준으로 학습한다?
    max_steps=500,
    # 이건 아래처럼 바뀌고
    # evaluation_strategy="steps",
    eval_strategy="steps",
    # 로그 저장 단위?가 steps, 50 step마다 저장
    logging_strategy="steps",
    logging_steps=50,
    # 로그는 여기다...
    # logging_dir="/data/logs",
    save_strategy="steps",
    save_steps=50,
    output_dir="/data/ckpt",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # 이것도 아래처럼 바뀌고
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=default_data_collator,
)

# 이건 텐서보드라는 걸 사용한다는데 어떻게 하는질 모르겠네...
# import subprocess

# pc = ["powershell", "-Command", "tensorboard --logdir=data/logs"]
# result = subprocess.run(pc, capture_output=True, text=True)
# print(result.stdout)

trainer.train()

# tensorboard --logdir=./logs

output_dir = "data/trained_model"
trainer.save_model(output_dir)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(output_dir)

print(tokenizer)
print(model)
