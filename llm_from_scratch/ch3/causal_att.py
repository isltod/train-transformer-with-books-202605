from scaled_dot_att_cls import SelfAttention_v2
import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)
# 3차원 벡터를 2차원 벡터로 바꾸기...
d_in = inputs.shape[1]  # 입력 임베딩 크기, d=3
d_out = 2  # 출력 임베딩 크기, d=2
torch.manual_seed(789)

# SelfAttention_v2 이용해서 가중치 준비
sa_v2 = SelfAttention_v2(d_in, d_out)

# 주대각 위를 0으로 마스킹하기 위해서 별도로 중요도 점수 α까지 계산
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 마스킹 행렬 만들고
context_length = attn_scores.shape[0]
# 주대각 위를 0으로 만드는 함수가 tril - tri lower인 듯...
mask_simple = torch.tril(
    torch.ones(context_length, context_length)
)  # diagonal=-1, triu()
print(mask_simple)

# 곱해서 어텐션 점수를 주대각 위는 0으로 마스킹...
# 그냥 torch.tril(attn_weights) 해도 되는데...
masked_simple = attn_weights * mask_simple
print(masked_simple)

# 마스킹 된 부분 제외하고 다시 합이 1이 되도록 정규화...
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
print(masked_simple_norm.sum(dim=-1, keepdim=True))

# 좀 더 효율적으로 상삼각 부분을 -inf로 만들어서 바로 소프트맥스에 적용한다...
# tri upper 함수로 상삼각에 1을 채우고
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# 1 = true이므로 그 자리에 -inf 채우기...
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# 그걸 소프트 맥스에 넣는데...-inf의 exp가 0이므로 같은 효과...
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 근데 결국은 둘이 단계 수는 같은거 같은데...아무튼 아래 방법이 더 세련되긴 하네...
