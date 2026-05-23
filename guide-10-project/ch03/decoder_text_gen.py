import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append("../../")
from wolf import get_my_gpu_device

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# 먼저 토치 데이터셋 상속받아서...
class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.block_size = block_size
        self.tokenizer = tokenizer

        with open(file_path, "r") as f:
            self.data = f.read()

        self.examples = []
        # block_size 만큼씩 끊어서
        for i in range(0, len(self.data) - self.block_size, self.block_size):
            example = self.data[i : i + self.block_size]
            # 토큰화하고, 텐서 리스트로 저장
            tokenized = self.tokenizer(
                example,
                padding="max_length",
                truncation=True,
                max_length=block_size,
                return_tensors="pt",
            )
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 어디서도 input_ids나 attention_mask로 사전을 안 만들었는데...그냥 되나?
        # 크기 1인 차원 제거하고 반환
        input_ids = self.examples[idx]["input_ids"].squeeze()
        attention_mask = self.examples[idx]["attention_mask"].squeeze()
        return input_ids, attention_mask


filename = "../../data/tiny_shakespeare_input.txt"
train_dataset = ShakespeareDataset(filename, tokenizer)


def collate_fn(batch):
    inputs, masks = zip(*batch)
    # (배치, 문장)으로 쌓고, (문장, 배치)로 변경해서 반환
    inputs = torch.stack(inputs).transpose(0, 1)
    masks = torch.stack(masks)
    return inputs, masks


# 토치 dataloader에도 collate_fn 지정하는 부분이 있었네...배치는 4...
train_dataloader = DataLoader(
    train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
)
# 데이터의 차원 확인 - (block_size, batch_size)인 (128, 4)로 나오겠지...
item = next(iter(train_dataloader))
input_ids, attention_masks = item
print(input_ids.shape, attention_masks.shape)


# 일단 이건 encoder_text.cls에서 했던 것과 같다고...
class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, dropout=0.1, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        postional_encoding = torch.zeros(max_seq_len, dim_embedding)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denom_term = torch.exp(
            torch.arange(0, dim_embedding, 2).float()
            * (-math.log(10000.0) / dim_embedding)
        )
        postional_encoding[:, 0::2] = torch.sin(position * denom_term)
        postional_encoding[:, 1::2] = torch.cos(position * denom_term)
        postional_encoding = postional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("postional_encoding", postional_encoding)

    def forward(self, x):
        x = x + self.postional_encoding[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, dropout):
        super().__init__()

        self.memory_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.memory_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # tgt는 target이라니, 이건 목표 임베딩과 목표 위치 인코더인데...
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tgt_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # 디코더에 트랜스포머가 들어가고...
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=dropout
            ),
            num_layers=num_layers,
        )

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.d_model = embedding_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1

        # 임베딩 층 초기화
        nn.init.uniform_(self.memory_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.tgt_embedding.weight, -initrange, initrange)

        # 디코딩 층 초기화
        for param in self.decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # 출력 층 초기화
        nn.init.uniform_(self.fc.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        tgt,
        memory=None,  # (문장 길이, 배치 크기)의 학습 데이터
        tgt_mask=None,  # 입력 시퀀스에 대한 마스킹
        memory_mask=None,  # 인코더 출력 시퀀스에 대한 마스
        memory_key_padding_mask=None,  # 인코더 출력 시퀀스에서 패팅 토큰 마스킹
        tgt_key_padding_mask=None,  # 입력 시퀀스에서 패딩 토큰 마스킹
    ):
        # 타겟 시퀀스는 입력 시퀀스를 한 칸씩 이동시킨 거라고...
        # 임베딩 차원 제곱근으로 스케일링 - 기울기 폭발/소멸 완화
        tgt = self.tgt_embedding(tgt) * self.d_model**0.5
        tgt = self.tgt_pos_encoder(tgt)
        print(tgt)
        memory = self.memory_embedding(memory) * self.d_model**0.5
        memory = self.memory_pos_encoder(memory)
        print(memory)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        print(output)
        output = self.fc(output)
        return output


device = get_my_gpu_device(0)


def generate_square_subsequent_mask(sz):
    # 행렬 받아서, 그 상삼각 부분은 -inf, 나머지 부분은 0인 행렬 만들어 반환...
    # upper triangular 행렬 부분을 반환...
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.to(device)


def create_mask(src, tgt, tokenizer_src=tokenizer, tokenizer_tgt=tokenizer):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 책 62쪽
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    # 책 63쪽
    src_padding_mask = (src == tokenizer_src.pad_token_id).transpose(0, 1)
    tgt_padding_mask = (tgt == tokenizer_tgt.pad_token_id).transpose(0, 1)
    return (
        src_mask.to(device),
        tgt_mask.to(device),
        src_padding_mask.to(device),
        tgt_padding_mask.to(device),
    )


model = TransformerDecoder(
    vocab_size=tokenizer.vocab_size, embedding_dim=768, num_layers=3, dropout=0.1
).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 여기서 학습 코드를 직접 작성하는 연습을 하라는데...시간 아깝다...
