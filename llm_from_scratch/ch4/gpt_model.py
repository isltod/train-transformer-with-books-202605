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


def generate_text_simple(model, idx, max_new_tokens, context_size):

    # 맨 처음 idx는 현재 문장이 담긴 (batch, n_tokens) 크기의 인덱스 배열로 시작

    for _ in range(max_new_tokens):
        # 현재 문장이 모델이 지원하는 문맥 크기를 초과하면 잘라냅니다.
        # 예를 들어, LLM이 5개 토큰만 지원하고 입력 문장의 크기가 10이라면,
        # 마지막 5개 토큰만 문맥으로 사용합니다.
        idx_cond = idx[:, -context_size:]

        # 예측을 만듭니다.
        with torch.no_grad():
            logits = model(idx_cond)

        # 문장의 마지막 단어만 뽑아서 사용 (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 확률을 얻기 위해 소프트맥스를 적용합니다.
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 가장 높은 확률 값을 가진 항목의 인덱스를 얻습니다. - 이 방식은 greedy decoding이라고...
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 선택한 인덱스를 현재 시퀀스에 추가합니다. (배치, 토큰 ID들)의 토큰 ID들 차원 마지막에 예측한 ID 추가
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


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

    # 간단한 텍스트 생성 예제
    start_context = "Hello, I am"
    # ID 묶음으로 바꾸고
    encoded = tokenizer.encode(start_context)
    print("인코딩된 ID:", encoded)
    # 배치 차원 추가하고
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 여기서는 실제로 드롭아웃이 있으니 그걸 끈다...
    model.eval()
    # ID 묶음으로 된 텐서를 넘겨 모델이 순차적으로 후속 단어를 하나씩 생성...
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    # 결과는 단어 ID들이 될테고...max_new_tokens=6 이니까 총 10개가 나올거고,
    print("출력:", out)
    print("출력 길이:", len(out[0]))

    # 그걸 다시 단어로 바꾸기 - 학습되질 않았으니 문장은 엉망인게 당연...
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
