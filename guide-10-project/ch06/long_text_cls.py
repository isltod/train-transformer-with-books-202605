# 뭐지? 이 병신같은 상황은? 이걸 맨 위에서 안하면 다 아무런 오류 메시지 없이 그냥 끝나버려?
# 그냥 됐다 안됐다 하는 개 병신 같은 상황이네...
import os.path
import sys

from transformers import AutoModelForSequenceClassification, pipeline
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

# IMDb 데이터셋 불러오기
imdb_data = load_dataset("stanfordnlp/imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.pad_token = tokenizer.eos_token

import nltk


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_sentence_length=48):
        self.data = data
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        # 단락을 문장 단위 리스트로 분할
        sentences = nltk.sent_tokenize(text)
        # 그걸 다시 단어 id 리스트로 바꾸고
        input_ids = [
            tokenizer.encode(
                sentence,
                max_length=self.max_sentence_length,
                truncation=True,
                padding="max_length",
            )
            for sentence in sentences
        ]
        attention_masks = [
            [1 if token_id != tokenizer.pad_token_id else 0 for token_id in sentence]
            for sentence in input_ids
        ]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 데이터를 데이터셋으로 만들고 그 중 10%만 학습에 적용
train_dataset = IMDBDataset(imdb_data["train"], tokenizer)
test_dataset = IMDBDataset(imdb_data["test"], tokenizer)

train_10_percent = int(len(train_dataset) * 0.1)
test_10_percent = int(len(test_dataset) * 0.1)
train_90_percent = len(train_dataset) - train_10_percent
test_90_percent = len(test_dataset) - test_10_percent

# 데이터셋을 10:90으로 분할
train_data_10_percent, _ = random_split(
    train_dataset, [train_10_percent, train_90_percent]
)
test_data_10_percent, _ = random_split(test_dataset, [test_10_percent, test_90_percent])


def pad_collate(batch):
    # 배치 안에 단락들이 있고, 그 안에 문장들이 있는데, 문장 수가 가장 많은 경우...
    max_num_sentences = max([item["input_ids"].shape[0] for item in batch])

    input_ids_batch = []
    attention_masks_batch = []

    # 배치별로 루프 실행. 만약 batch_size=4이면 4회 루프 실행
    for item in batch:
        # 최대 문장 수만큼 패딩
        num_sentences = item["input_ids"].shape[0]
        pad_length = max_num_sentences - num_sentences
        input_ids = torch.cat(
            [
                item["input_ids"],
                torch.zeros(pad_length, item["input_ids"].shape[1], dtype=torch.long),
            ],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                item["attention_mask"],
                torch.zeros(
                    pad_length, item["attention_mask"].shape[1], dtype=torch.long
                ),
            ],
            dim=0,
        )
        # 문장 수까지 맞춘 데이터를 리스트에 이어 붙이고...
        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_mask)

    # 리스트 원소들을 쌓아서 (배치, 문장 수, 문장 길이) 텐서로...
    input_ids_tensor = torch.stack(input_ids_batch, dim=0)
    attention_masks_tensor = torch.stack(attention_masks_batch, dim=0)
    labels_tensor = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_masks_tensor,
        "label": labels_tensor,
    }


bs = 4
train_loader = DataLoader(
    train_data_10_percent, batch_size=bs, shuffle=True, collate_fn=pad_collate
)
test_loader = DataLoader(
    test_data_10_percent, batch_size=bs, shuffle=True, collate_fn=pad_collate
)
item = next(iter(train_loader))
print("input_ids", item["input_ids"].shape)
print("attention_mask", item["attention_mask"].shape)
print("label", item["label"].shape)


class FineTunedBertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(FineTunedBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids_list, attention_mask_list):
        batch_pooled_outputs = []

        # 이건 input_ids_list에 대한 루프고...아마도 단락?
        for batch_input_ids, batch_attention_mask in zip(
            input_ids_list, attention_mask_list
        ):
            pooled_outputs = []

            # 이건 그 안에 input_ids에 대한 루프...즉 문장에 대한 루프
            # pooler_output은 sentence 표현형(representation)임.
            for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
                output = self.bert(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                # bert의 경우 output['pooler_output']은 sentence 임베딩이라고...
                pooler_output = output["pooler_output"]
                pooled_outputs.append(pooler_output)

            # pooled outputs를 단일 텐서로 스태킹
            pooled_outputs_tensor = torch.stack(pooled_outputs, dim=1)

            # 첫 번째 차원(dim=1)을 따라 평균 계산
            concatenated_output = torch.mean(pooled_outputs_tensor, dim=1)
            batch_pooled_outputs.append(concatenated_output)

        # batch_pooled_outputs을 단일 텐서로 스태킹
        batch_pooled_outputs_tensor = torch.stack(batch_pooled_outputs, dim=0)

        output = self.dropout(batch_pooled_outputs_tensor)

        output = torch.squeeze(output, dim=1)

        return self.fc(output)


print(len(train_loader))

import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm

accelerator = Accelerator(gradient_accumulation_steps=bs * 2, mixed_precision="bf16")
# accelerator = Accelerator()
device = accelerator.device

model = FineTunedBertClassifier(n_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

model, optimizer = accelerator.prepare(model, optimizer)

criterion = nn.CrossEntropyLoss()

train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
# 모델을 100회 배치마다 저장
save_interval = 100

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loader_progress = tqdm(
        enumerate(train_loader),
        desc=f"Epoch {epoch + 1}/{num_epochs}, Training",
        total=len(train_loader),
    )
    for batch_idx, batch in train_loader_progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # optimizer.zero_grad()
        # logits = model(input_ids, attention_mask)
        # loss = criterion(logits, labels)
        # accelerator.backward(loss)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()

        if batch_idx % 10 == 0:
            train_loader_progress.set_postfix(loss=loss.item())
        if (batch_idx + 1) % save_interval == 0:
            model_save_path = f"../../data/model/AlbertTextClassifier_epoch{epoch + 1}_batch{batch_idx + 1}.pt"
            torch.save(model.state_dict(), model_save_path)
    model.eval()
    total_correct = 0
    total_samples = 0
    test_loader_progress = tqdm(
        test_loader, desc=f"Epoch {epoch + 1}/{num_epochs}, Testing"
    )

    with torch.no_grad():
        for batch in test_loader_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}")
