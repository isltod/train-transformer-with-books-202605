GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 어휘사전 크기
    "context_length": 1024,  # 문맥 길이
    "emb_dim": 768,  # 임베딩 차원
    "n_heads": 12,  # 어텐션 헤드 개수
    "n_layers": 12,  # 층 개수
    "drop_rate": 0.1,  # 드롭아웃 비율
    "qkv_bias": False,  # 쿼리, 키, 값을 만들 때 편향 포함 여부
}
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 토큰 임베딩은 단어사전 룩업이니까, (사전, 단어표현) 차원
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 위치 임베딩은 문장 내 단어들에 꼬리표니까 (단어순서, 단어표현) 차원
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 나중에 TransformerBlock이 될 자리에 아무것도 안하는 Module 들을 미리 배치...
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 이것도 나중에 층 정규화라는 것을 할 자리에 아무것도 않하는 Module 배치
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 마지막에 소프트맥스 입력 값을 만드는 완전연결 층
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 배치 개의 문장(단어 ID 묶음)이 들어오면...(2,4)에서 4가 50257까지 ID
        batch_size, seq_len = in_idx.shape
        # 단어 임베딩하고 (2,4,768)에서 768이 50257까지 ID로 만든 768차원 단어표현 벡터
        tok_embeds = self.tok_emb(in_idx)
        # 문장 내 단어 수에 맞춰 절대 위치 임베딩 만들고 (4,768)에서 4는 그냥 0~3
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 더해서 임베딩 완성...브로드 캐스팅으로 (2,4,768)
        x = tok_embeds + pos_embeds
        # 드롭아웃 (2,4,768)
        x = self.drop_emb(x)
        # 트랜스포커 블록(셀프 어텐션) 거치고...우선 아무 처리 안하니 (2,4,768)
        x = self.trf_blocks(x)
        # 층 정규화라는 걸 하고...이것도 아무 처리 안하니 (2,4,768)
        x = self.final_norm(x)
        # 결과를 내는데 소프트 맥스에 넣을 로짓으로 내는 모양...(2,4,768) @ (2,768,50257) -> (2,4,50257)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 더미 클래스

    def forward(self, x):
        # 이 블록은 아무것도 하지 않고 입력을 그냥 반환합니다.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 층 정규화 인터페이스를 흉내내기 위한 매개변수

    def forward(self, x):
        # 이 블록은 아무것도 하지 않고 입력을 그냥 반환합니다.
        return x


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("출력 크기:", logits.shape)
print(logits)
