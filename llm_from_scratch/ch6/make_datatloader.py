from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import tiktoken
import pandas as pd

tokenizer = tiktoken.get_encoding("gpt2")
# <|endoftext|> 토큰 ID 확인 50256
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # 데이터는 Label, Text 컬럼으로 되어있고...
        self.data = pd.read_csv(csv_file)
        # 그 중 Text 컬럼 데이터로 텍스트 토큰화
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # max_length 없으면 가장 긴 문장에 맞춰 패딩...
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            # 있으면 그 길이로 자르고 모자라면 패딩...
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # 최대 길이에 맞춰 패딩하기
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    # 가장 긴 문장 길이 반환
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# 바로 전 prepare_spam_ds에서 만들어뒀던 훈련/검증/시험 데이터 읽어서 데이터셋으로...
train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
print(train_dataset.max_length)
# 검증과 시험 데이터셋은 max_length를 훈련에 맞춰준다...
val_dataset = SpamDataset(
    csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)

num_workers = 0
batch_size = 8

torch.manual_seed(123)
# 훈련/검증/시험 데이터셋으로 데이터로더 만들기...
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

print("훈련 세트 로더:")
# 밑에서 input_batch 등을 사용하려고 만든 껍데기 for문...
for input_batch, target_batch in train_loader:
    pass
print("입력 배치 차원:", input_batch.shape)
print("레이블 배치 차원", target_batch.shape)
print(f"{len(train_loader)}개 훈련 배치")
print(f"{len(val_loader)}개 검증 배치")
print(f"{len(test_loader)}개 테스트 배치")
