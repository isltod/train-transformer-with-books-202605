import tiktoken
import torch
from previous_chapters import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 어휘 사전 크기
    "context_length": 256,  # 짧은 문맥 길이 (원본 길이: 1024)
    "emb_dim": 768,  # 임베딩 차원
    "n_heads": 12,  # 어텐션 헤드 개수
    "n_layers": 12,  # 층 개수
    "drop_rate": 0.1,  # 드롭아웃 비율
    "qkv_bias": False,  # 쿼리-키-값 생성시 편향 사용 여부
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
# 추론 시에는 드롭아웃을 비활성화합니다
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 배치 차원을 추가합니다.
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 배치 차원을 삭제합니다
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))
