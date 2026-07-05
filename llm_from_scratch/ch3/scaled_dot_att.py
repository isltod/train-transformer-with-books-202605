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

# 역시 우선 간단하게 journey 만 가지고...
x_2 = inputs[1]  # 두 번째 입력 원소
# 3차원 벡터를 2차원 벡터로 바꾸기...
d_in = inputs.shape[1]  # 입력 임베딩 크기, d=3
d_out = 2  # 출력 임베딩 크기, d=2

# query, key, value 벡터를 만들(투영 할) 행렬 3가지 초기화...실제로 학습되는 것들은 바로 이거!
torch.manual_seed(123)
# requires_grad=False는 출력 간단하게...실제로는 True로 해야된다..
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 행렬을 곱해서 쿼리, 키, 값 벡터 생성
query_2 = x_2 @ W_query  # 두 번째 입력 원소에 대한 값을 계산하므로 _2로 씁니다.
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# 1 단계: 쿼리, 키, 값 벡터 만들기
# 전체 벡터들에 대해서 적용하면 6개의 임베딩 벡터가 3차원에서 2차원으로 투영된다...
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 2 단계: 어텐션 점수 ω 계산 - 단어끼리가 아니라, 쿼리와 키로 바꾼 벡터 사이에 닷곱
keys_2 = keys[1]  # 파이썬 인덱스는 0부터 시작합니다.
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
# 기초에서 했듯이 쿼리를 모든 단어들(키)에 닷곱...벡터가 앞으로 나오면서 행렬 전치...
attn_scores_2 = (
    query_2 @ keys.T
)  # 주어진 쿼리에 대한 모든 어텐션 점수 (1,2)x(2,6) = (1,6)
print(attn_scores_2)

# 3단계: 중요도 점수 α 계산 - 여기서 키 임베딩 차원의 제곱근으로 스케일링
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# 4단계: 컨텍스트 벡터를 만든다 - 이것도 α에 단어를 곱하는게 아니라 value 곱
# journey에 대한 컨텍스트 벡터...(1,6) x (6,2) = (1,2)
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
