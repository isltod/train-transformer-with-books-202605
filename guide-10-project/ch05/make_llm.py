import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from accelerate import Accelerator
import re
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
import random
from torch.optim import AdamW
from transformers import BertForPreTraining, BertConfig
import sys

sys.path.append("../../")
from wolf import Timer

# # 처음에 한 번은 실행되야 한다...
# import nltk
#
# nltk.download("punkt_tab")


def create_sentence_dataframe(df):
    # 알파벳 외 특수 문자는 제거 정규식
    special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s.,?!]+|\n")

    sentences = []
    # 데이터프레임의 각 행 기준으로 반복 루프
    for text in df["TEXT"]:
        # 특수 문자 제거하고 토큰화해서 리스트로 묶기
        clean_text = special_chars_pattern.sub("", text)
        tokenized_sentences = sent_tokenize(clean_text)
        # 토큰 자체도 리스트라서 원소들만 추가하려면 append 말고 extend
        sentences.extend(tokenized_sentences)

    # 정리한 문장(sentences)을 담은 데이터프레임 생성
    sentence_df = pd.DataFrame(sentences, columns=["text"])

    return sentence_df


# medical_data.csv
data_txt = pd.read_csv("../../data/medical_data.csv")
pd.options.display.max_colwidth = 100
data = create_sentence_dataframe(data_txt)
print(data.shape)
print(data.head())


# 사용자 정의 데이터셋
class ClinicalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # BERT 학습은 문장 "1 [SEP] 문장 2" 형식으로 반환해야한다...
        # 첫 번째 문장은 받은 인덱스 문장
        news = self.data.loc[idx, "text"]
        if idx + 1 < len(self.data):
            # 데이터셋에서 뒤에 한 문장 더 있다면 그걸 두 번째 문장으로...
            next_news = self.data.loc[idx + 1, "text"]
        else:
            # 없다면 맨 앞의 문장을 두 번째 문장으로...
            next_news = self.data.loc[0, "text"]

        # 스페셜 토큰 추가하고 토큰화
        combined_news = news + " [SEP] " + next_news
        tokenized = self.tokenizer(
            combined_news,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 반환 형식은 딕셔너리로 묶어서...
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "text": combined_news,
        }


# 토크나이저 생성해서 데이터셋 만들기...
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = ClinicalDataset(data, tokenizer)
print(dataset[0])
print(tokenizer.sep_token_id)


class DataCollatorForPreTraining(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, nsp_probability=0.5):
        # MLM Masked Language Modeling 사용한다는 말이겠지? 비율은 15%?
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        # 이어진 문장이 다음 문장일 확률을 50%로 맞추는게 목표라고...
        self.nsp_probability = nsp_probability

    # 사전 훈련을 위해 이걸 오버라이딩?
    def __call__(self, examples):
        # Next Sentence Prediction 라벨
        nsp_labels = []
        input_ids_list = []
        attention_masks_list = []
        labels_list = []

        # 문장쌍마다 돈다는 얘기겠지?
        for example in examples:
            # 간단한 건 먼저 받고
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            if random.random() > self.nsp_probability:
                # 이 확률로 맞는 문장쌍...라벨
                nsp_labels.append(1)
            else:
                # 안맞는 확률...먼저 라벨
                nsp_labels.append(0)

                # 안맞게 만들기 위해 뒤 문장 토큰들을 섞는다...그냥 정상적인 다른 문장을 쓰지 않고?
                # 먼저 토큰들 id가  SEP id인 경우를 텐서로...
                is_sep = input_ids == self.tokenizer.sep_token_id
                # 0 아닌 요소 인덱스가 바로 SEP의 인덱스...그걸 행렬 텐서로 받으면 불편하니 튜플로 받고 숫자만...
                sep_idx = is_sep.nonzero(as_tuple=True)[0][0].item()
                # 그럼 두 번째 문장은 SEP 이후...
                second_sentence = input_ids[sep_idx + 1 :]
                # 그걸 섞는다...
                second_sentence = second_sentence[
                    torch.randperm(second_sentence.size()[0])
                ]

                # 멀쩡한 첫 번째와 뒤섞은 두 번째 문장을 다시 묶어주면 False 데이터 완성
                input_ids = torch.cat(
                    (input_ids[: sep_idx + 1], second_sentence), dim=0
                )

            # 그렇게 id 토큰들, 마스크 추가...
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

            # 뭔진 모르겠는데, 앞 문장만 -100으로 넣고 그걸 라벨이라고?
            sep_idx = (
                (input_ids == self.tokenizer.sep_token_id)
                .nonzero(as_tuple=True)[0][0]
                .item()
            )
            labels = input_ids.clone()
            labels[sep_idx:] = -100
            # 암튼 이걸 중간중간 마스킹을 위해 상속받은 클래스의 __call()__을 이용하는데 넣는다...
            labels_list.append(labels)

        # 상속받은 DataCollatorForLanguageModeling 클래스는 이런 딕셔너리 리스트가 필요한 모양...
        example_dicts = [
            {"input_ids": ids, "attention_mask": mask, "labels": lbl}
            for ids, mask, lbl in zip(input_ids_list, attention_masks_list, labels_list)
        ]

        # 부모 클래스를 사용해서 MLM 처리
        batch = super().__call__(example_dicts)

        # 배치에 NSP 레이블 추가
        batch["next_sentence_label"] = torch.tensor(nsp_labels, dtype=torch.long)

        return batch


# 토크나이저, 데이터셋, 데이터 콜레이터(collator) 초기화
data_collator = DataCollatorForPreTraining(tokenizer)

# DataLoader 생성
train_dataloader = DataLoader(
    dataset, shuffle=True, collate_fn=data_collator, batch_size=4
)
item = next(iter(train_dataloader))
print(len(train_dataloader))
print("ids", item["input_ids"][0])
print("mask", item["attention_mask"][0])
print("labels", item["labels"][0])
print("next_sentence_label", item["next_sentence_label"][0])

accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")

# BERT 모델 불러오기
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertForPreTraining(config)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 모델과 옵티마이저를 accelerate.prepare()로 처리
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

num_epochs = 1
print_every = 100
with Timer():
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            next_sentence_label = batch["next_sentence_label"].to(accelerator.device)

            with accelerator.accumulate(model):
                # 포워드 패스(pass)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    next_sentence_label=next_sentence_label,
                )
                loss = outputs.loss

                # 백워드 패스(pass)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

            if (step + 1) % print_every == 0:
                print(f"Step {step + 1}: Loss = {running_loss / print_every:.4f}")
                running_loss = 0.0

print("Training complete!")
# 원서와 달리 다음 구글 드라이브에 모델을 저장
save_directory = "../../data/pretrained_bert/"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
