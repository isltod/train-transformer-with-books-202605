import matplotlib.pyplot as plt
import tiktoken
import torch
from previous_chapters import GPTModel, generate_text_simple
from gen_text_simple import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text

# 반면에 옵티마이저와 같이 저장된 모델을 다시 로드할 때는 load_state_dict 대신 load
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
# 그리고 사전 키를 이용해서 가중치와 옵티마이저 따로 지정...
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# NEW: 이후 코드의 결과를 책과 일치하도록 만들기 위해 CPU를 사용합니다.
inference_device = torch.device("cpu")

model.to(inference_device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))
# 훈련데이터가 적어서 몇 번 안돌려도 모델이 텍스트를 통채로 외워서 답을 낸다?

# 모델 출력 로짓을 확률 분포로 보고 샘플링하는 온도 샘플링...
# 예제를 위해 아주 간단한 단어 사전...
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

# "every effort moves you"라고 입력했더니 LLM이 다음 토큰(단어 하나)을 위해 아래와 같은 로짓을 반환했다고 가정해
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
# 거기 확률은 당연히 소프트맥스로 구하고...
probas = torch.softmax(next_token_logits, dim=0)
# 소위 그리디 샘플링이라는 최고값만 보면 항상 같은 결과를 내는데...
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
# multinomial을 쓰면 일단 확률적 샘플링을 한다..
next_token_id = torch.multinomial(probas, num_samples=1).item()
# 근데 결과는 마찬가지...forward가 약 60% 정도 확률이라서 잘 나오기 때문에...
print(inverse_vocab[next_token_id])


# 그걸 확인하기 위해서 1000번 돌리고 빈도 출력
def print_sampled_tokens(title, probas):
    torch.manual_seed(123)  # 재현가능성을 위한 랜덤 시드
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    # 정수로 된 1차원 텐서 내에서 0, 1, 2...가 각각 몇개인지 텐서로 반환
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    print(title)
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


print_sampled_tokens("multimodal로 1000번 샘플링:", probas)


# 온도 샘플링으로 선택 가능성 조절...temperature는 양수...
def softmax_with_temperature(logits, temperature):
    # 1보다 큰 수로 나누면 분포가 좀 더 고르게 변해서 다른 단어 고를 확률 증가,
    # 1보다 작은 수로 나누면 분포가 좀 더 뾰족해져서 argmax와 비슷해져 가고...
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# 온도 값
temperatures = [1, 0.1, 5]  # 원본, 낮은 온도, 높은 온도

# 스케일을 조정한 확률 계산
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

# 그렇게 온도로 조절한 확률을 그래프로...
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
# 위에 세 가지 온도별로 돌면서
for i, T in enumerate(temperatures):
    # x축 위치는 원래 x 축 위치에 반복수만큼 우측 이동...
    rects = ax.bar(
        x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
    )

ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.show()

# 온도 5로 스케일링한 경우 빈도가 0인 것들이 사라지고 비슷해져간다..
print_sampled_tokens("temp 5 scaling:", scaled_probas[2])

# top k 샘플링 - 결국 확률 높은 몇 개로 한정한다는 얘기...
top_k = 3
# 특정 차원을 기준으로 가장 큰 k개의 값을 찾아서 값과 인덱스 반환
top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("탑-k 로짓:", top_logits)
print("탑-k 위치:", top_pos)

# 토큰 리스트와 같은 모양의 -inf 텐서 만들고
new_logits = torch.full_like(next_token_logits, -torch.inf)
# top k 위치에 그 값 할당...
new_logits[top_pos] = next_token_logits[top_pos]
topk_probas = torch.softmax(new_logits, dim=0)
print("top k 적용 후 확률\n", topk_probas)


# temp scaling과 top k 적용한 텍스트 생성 함수...
def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    # 생성할 단어 수만큼 돌면서...
    for _ in range(max_new_tokens):
        # idx는 (배치, 각 문장의 토큰 ID들) 텐서...받은 문장들이 위치 임베딩 길이보다 크면 잘라내고...
        idx_cond = idx[:, -context_size:]
        # 역전파 끊고 예측 받아서, 마지막 단어에 대한 예측만 뽑기...
        with torch.no_grad():
            logits = model(idx_cond)
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 1. 탑-k 샘플링으로 로짓을 필터링합니다.
        if top_k is not None:
            # 가장 큰 k 값 뽑아서
            top_logits, _ = torch.topk(logits, top_k)
            # 그 중 가장 작은 값
            min_val = top_logits[:, -1]
            # 그 값을 기준으로 -inf 마스킹
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # 2. 온도 스케일링을 적용합니다.
        if temperature > 0.0:
            logits = logits / temperature

            # (책에 없음): mps 장치에서 동일한 결과를 얻기 위해 수치 안정성을 위한 팁
            # 소프트맥스 전에 행의 최댓값을 뺍니다.
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # 소프트맥스 함수를 적용하여 확률을 얻습니다.
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 분포에서 샘플링합니다.
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 온도 스케일링을 사용하지 않는 경우 이전처럼 그리디 샘플링을 사용해 다음 토큰을 선택합니다.
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # eos_id가 지정되어 있고 EoS 토큰을 만나면 생성을 중단합니다.
        if idx_next == eos_id:
            break

        # 이전과 동일하게 샘플링된 인덱스를 현재 시퀀스 뒤에 추가합니다.
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)

print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))
# 좀 다른 문장이 출력되는 효과는 있는데 그래도 문장이 별로라면 별 의미 없을 듯...
