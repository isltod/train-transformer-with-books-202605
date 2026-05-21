# 원래 모델과 토크나이저는 쌍을 이뤄야 한다고...
# 그래서 my_tokenizer.py에서 만든 토크나이저를 위해서 모델을 초기화하고 학습시킨다....
from datasets import load_dataset
from transformers import BertTokenizerFast

path = "e:/Devs/train-transformer-with-books-202605/"
dataset = load_dataset("klue", "ynat")
model_name = path + "data/MyTokenizer"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

from transformers import BertConfig

# 모델의 최초 선언을 위해서 config가 필요하다?
cfg = BertConfig
print(cfg)

# config에는 임베딩 크기, 히든 크기, 레이어 수 등 모델 구조가 들어가는데 다 기본으로...
# 사전 크기만 만들어 놓은 토크나이저 따라서 설정...
mycfg = BertConfig(vocab_size=tokenizer.vocab_size)

from transformers import BertForMaskedLM

# 모델도 적당한 클래스 갖다가 생성저에 config 넣고 생성하면 새로 만들어진다고...
model = BertForMaskedLM(mycfg)
# 구조는 이렇게 볼 수 있다고...
print(model.config)

# 3장 말미에 했던 것처럼, dataset.map, TrainingArguments, collator, Trainer 만들어서 학습...후 저장...
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

datasets = dataset.map(
    lambda x: tokenizer(x["title"]),
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names["train"],
)

# 그럼 이게 문제일테니 3장 참고해서 수정한다...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU만 사용
os.environ["TENSORBOARD_LOGGING_DIR"] = path + "data/logs"
args = TrainingArguments(
    # 계속 죽다가 여기부터 아래까지 바꾸니 돌아 간다...8도 된다...결국 gra.. 아니면 fp 설정이 도움이 된다는 얘기...
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=8,  # 배치 사이즈 대신 누적 스텝 활용
    gradient_checkpointing=True,  # Gradient Checkpointing 활성화
    fp16=True,  # 혼합 정밀도 학습 (메모리 절약)
    # 여기까지....
    learning_rate=5e-5,
    max_steps=1000,
    # 이건 아래처럼 바뀌고
    # evaluation_strategy="steps",
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=100,
    # 로그 경로 설정은 위에 환경변수 설정으로 변경...
    # logging_dir="/data/logs",
    output_dir=path + "data/ckpt",
    report_to="tensorboard",
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    # 이것도 아래처럼 바뀌고
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=collator,
)

trainer.train()
trainer.save_model(path + "data/MyBertModel")
