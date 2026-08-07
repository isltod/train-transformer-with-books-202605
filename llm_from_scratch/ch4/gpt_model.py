import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from layer_norm import LayerNorm
from dummy_gpt import GPT_CONFIG_124M
import tiktoken
from transformers import GPT2Tokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 책 156 쪽 구조
        # 근데 Linear 층은 W_T 구조로 저장하는데, Embedding 층은 그냥 W 구조로 저장하네...(50257, 768)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 이 드롭아웃은 트랜스포머 들어가기 전에 거치는 층
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 이게 핵심...트랜스포머 블록
        self.trf_blocks = nn.Sequential(
            # 이 블록이 모델 크기에 따라 여러번 반복...
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 학습 안정화와 관련 있다고..
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 마지막 완전연결은 편향 b 없이...여긴 W_T라서 (50257, 768)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 크기 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    # 허깅페이스 경고 처리 - 깃허브에 secret 올릴 수 없으니 .env 파일로 처리...
    # ..env 파일 불러오기
    load_dotenv()
    # 환경 변수에서 키 가져오기
    api_key = os.getenv("API_KEY")
    login(token=api_key)

    cfg = GPT_CONFIG_124M
    # # 161 연습문제, 임베딩 차원 1280, 트랜스포머 블록 36, 멀티헤드 어텐션 20 경우는?
    # cfg["emb_dim"] = 1280
    # cfg["n_layers"] = 36
    # cfg["n_heads"] = 20
    # # ------------------------------
    model = GPTModel(cfg)
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # 샘플 데이터
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # 문장 2 단어 8개의 ID 텐서 (2,4)
    print("입력 배치:\n", batch)

    # 훈련 안된 상태에서 그냥 예측
    out = model(batch)

    # 출력은 입력에 단어사전 룩업 차원 추가 (2,4,50257)
    print("\n출력 크기:", out.shape)
    print(out)

    # 163,009,536개
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 개수: {total_params:,}")
    # 1억 2천 4백만개가 아닌데, 토큰 임베딩과 출력 가중치가 뭔가 서로 연결된다는 거 같은데...그래서 이 둘이 같고...
    print("토큰 임베딩 층의 가중치 크기:", model.tok_emb.weight.shape)
    print("출력 층의 가중치 크기:", model.out_head.weight.shape)
    # 실제로 이 둘에 같은 가중치를 사용해서, 출력 가중치를 빼주만 1억 2천 4백만개...
    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(f"가중치 묶기를 고려한 훈련 가능한 파라미터 개수: {total_params_gpt2:,}")

    # 총 크기를 바이트 단위로 계산합니다(float32라 가정하면 파라미터당 4바이트입니다).
    total_size_bytes = total_params * 4
    # 다시 메가바이트로 변환합니다.
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"모델에 필요한 메모리 공간: {total_size_mb:.2f} MB")
