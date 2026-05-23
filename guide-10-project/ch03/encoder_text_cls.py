import math
import sys

sys.path.append("../../")
from wolf import get_my_gpu_device

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

# RTX 3090 GPU는 먼 잡아놓고 시작하자..
device = get_my_gpu_device(0)

# IMDB 데이터셋과 토크나이저 불러오기
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# 데이터셋 토큰화
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )


# 훈련과 검증 데이터셋은 토큰화 함수를 연결해서 생성...
train_dataset = dataset["train"].map(
    tokenize, batched=True, batch_size=len(dataset["train"])
)
val_dataset = dataset["test"].map(
    tokenize, batched=True, batch_size=len(dataset["test"])
)

# 토큰화된 데이터셋에서 훈련 데이터(input_ids)와 마스크(attention_mask), 정답지 추출
train_data = torch.tensor(train_dataset["input_ids"])
train_attention_mask = torch.tensor(train_dataset["attention_mask"])
train_labels = torch.tensor(train_dataset["label"])
# 검증 데이터도 마찬가지...
val_data = torch.tensor(val_dataset["input_ids"])
val_attention_mask = torch.tensor(val_dataset["attention_mask"])
val_labels = torch.tensor(val_dataset["label"])

# 그걸 TensorDataset으로 생성
train_dataset = TensorDataset(train_data, train_attention_mask, train_labels)
val_dataset = TensorDataset(val_data, val_attention_mask, val_labels)


# 뭔가 문장 길이 맞춰주는 거겠지?
def collate_fn(batch):
    # 암튼 데이터는 input_ids, attention_mask, labels 순서로 들어오는데...
    input_ids, attention_mask, labels = zip(*batch)
    input_ids = torch.stack(input_ids).transpose(
        0, 1
    )  # input_ids 트랜스포즈(Transpose)
    attention_mask = torch.stack(attention_mask)  # attention_mask 트랜스포즈(Transpose)
    # 왜 정답지만 gpu로 보내지?
    labels = (
        torch.nn.functional.one_hot(torch.tensor(labels), num_classes=2)
        .float()
        .to(device)
    )
    return input_ids, attention_mask, labels


# 데이터셋과 정리 함수로 데이터로더 만들기...
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)


# 내친김에 분류 클래스 생성 텍스트 분류 연습을 하는데, 그 전에 토치에는 없는 위치인코딩 클래스부터...
class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, dropout=0.1, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        postional_encoding = torch.zeros(max_seq_len, dim_embedding)
        # torch.arange로 [0,1,2...] 만들고 unsqueeze(1)로 [[1], [2]...]로 바꾸고...
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 분모라...뭐가 나오나?
        denom_term = torch.exp(
            torch.arange(0, dim_embedding, 2).float()
            * (-math.log(10000.0) / dim_embedding)
        )
        # 짝수는 사인, 홀수는 코사인
        postional_encoding[:, 0::2] = torch.sin(position * denom_term)
        postional_encoding[:, 1::2] = torch.cos(position * denom_term)
        # 0 축에 배치 만들고 0과 1을 바꾸나? 그냥 1 축에 만들면 되는 거 아녀?
        postional_encoding = postional_encoding.unsqueeze(0).transpose(0, 1)
        # 역전파 학습은 없고, 상태 저장하는 버퍼...를 postional_encoding으로..
        self.register_buffer("postional_encoding", postional_encoding)

    def forward(self, x):
        x = x + self.postional_encoding[: x.size(0), :]
        return self.dropout(x)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, num_classes):
        super(TextClassifier, self).__init__()

        # 시작은 임베딩
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 위에서 만든 커스텀 위치인코딩...
        self.positional_encoding = PositionalEncoding(embedding_dim)
        # 트랜스포머 인코더 층 생성 - 레이어를 먼저 만들고 거기에 끼워넣기...
        # 여기서 embedding_dim // nhead == 0이어야 한다...
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead)
        # 인코더 레이어를 실제로 인코더에 넣어 만든다..
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # 출력은 완전연결...
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        # 뭐...가중치 초기화겠지...
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, key_padding_mask=None):
        # 순서대로 임베딩, 위치인코딩, 트랜스포머 인코딩...
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # 평균 풀링 - 첫 번째 차원을 평균값으로 삭제
        # (문장 길이, 배치 크기, 임베딩 차원) -> (배치 크기, 임베딩 차원)
        x = x.mean(dim=0)

        # 분류 작업용 완전 연결 층
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


import torch.optim as optim  # optim 모듈 임포트 추가
import torch.nn as nn

vocab_size = tokenizer.vocab_size
embedding_dim = 512
nhead = 8
num_layers = 6
num_classes = 2

# 모델 생성
model = TextClassifier(vocab_size, embedding_dim, nhead, num_layers, num_classes).to(
    device
)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 시킨다...
num_epochs = 1
for epoch in range(num_epochs):
    i = 0
    for batch_data, batch_attention_mask, batch_labels in train_dataloader:

        optimizer.zero_grad()

        # attention_mask를 불리언(boolean) 텐서로 변환
        batch_attention_mask = (batch_attention_mask == 0).to(device)

        outputs = model(batch_data.to(device), key_padding_mask=batch_attention_mask)
        loss = criterion(outputs, batch_labels.to(device))
        if i % 100 == 0:
            print("epoch ", epoch, "batch ", i, "loss ", loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        i = i + 1

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# 일단 모델 저장하고...
torch.save(model.state_dict(), "../../data/TextClassificationModel.pth")

# 토크나이저 초기화...왜 초기화?
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
embedding_dim = 512
nhead = 8
num_layers = 6  # 원서 코드에서는 3으로 잘못 기재되어 고침
num_classes = 2

# 위에서 만든 클래스로 모델 전형을 생성
model_loaded = TextClassifier(
    vocab_size, embedding_dim, nhead, num_layers, num_classes
).to(device)

# 그리고 저장된 학습 모델 가중치(weights) 불러오기
model_loaded.load_state_dict(torch.load("../../data/TextClassificationModel.pth"))
print(model.eval())

# 근데 numel은 뭐냐?
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params)


# 주어진 텍스트에서 추론을 실행하는 함수
def infer(text):
    # 입력 텍스트 토큰화 - 여기도 tokenizer는 글로벌 변수를 막 갖다 쓰는 코드일테고...
    tokens = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    input_ids = tokens["input_ids"].to(device).transpose(0, 1)

    attention_mask = tokens["attention_mask"]
    attention_mask = (attention_mask == 0).to(device)
    print(input_ids.shape)
    print(attention_mask)

    # 추론 실행
    with torch.no_grad():
        output = model(input_ids, key_padding_mask=attention_mask)
    # 출력을 클래스 확률로 변환
    probs = output.squeeze(0)
    return probs


# 샘플 텍스트로 테스트
example_text = "This movie is  good! ."
probabilities = infer(example_text)

print("Probabilities:", probabilities)
