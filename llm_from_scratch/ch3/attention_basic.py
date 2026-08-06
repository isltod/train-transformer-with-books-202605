import torch

# 일단 이게 입력 문장이고...
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

# 먼저 journey 하나를 대상으로 전체적인 로직을 보면...

# 1: 어텐션 점수 ω 계산...
query = inputs[1]  # 두 번째 입력 토큰이 쿼리입니다
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # 닷곱 어텐션은 결국 단어 임베딩 사이의 닷곱...커뮤트하니 순서는 상관없고, 1차원 벡터 dot은 전치 필요없다고...
    attn_scores_2[i] = torch.dot(query, x_i)
print("어텐션 점수:", attn_scores_2)

# 2: 중요도 점수 α 계산 - 소프트맥스 처리...
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("어텐션 가중치:", attn_weights_2)
print("가중치 합:", attn_weights_2.sum())

# 3:컨텍스트 벡터를 만든다
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    # 이건 닷곱 아니고 스칼라 곱
    context_vec_2 += attn_weights_2[i] * x_i
print("x2에 대한 문맥 벡터:", context_vec_2)

# 이걸 입력 문장의 모든 토큰들에 대해서 하는데...

# 1: 어텐션 점수 ω 계산...앞이 행으로 나오니 쿼리겠지...
attn_scores = inputs @ inputs.T  # [6,3]@[3,6]=[6,6]
print(attn_scores)

# 2: 중요도 점수 α 계산 - 소프트맥스 처리...
# dim=-1은 마지막 차원에 대해서...마지막 차원에 있는 값들, 그러니까 가장 안쪽 괄호 안의 합이 1이 되도록...
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
print("두 번째 행:", attn_weights[1])
row_2_sum = sum(attn_weights[1])
print("두 번째 행의 합:", row_2_sum)
print("모든 행의 합:", attn_weights.sum(dim=-1))

# 3:컨텍스트 벡터를 만든다
# 이것도 뒤가 임베딩 벡터..각 행이 뒤 1~3열에 곱해서 결과의 그 행의 1~3열에 들어가 컨텍스트 벡터로...
all_context_vecs = attn_weights @ inputs  # [6,6]@[6,3] = [6,3]
print(all_context_vecs)
print("이전에 계산한 두 번째 문맥 벡터:", context_vec_2)
