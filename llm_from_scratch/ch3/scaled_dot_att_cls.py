import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        # query, key, value 벡터를 만들(투영 할) 행렬 3가지 초기화...
        # requires_grad는 기본값이 True인 모양...
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # 1 단계: 쿼리, 키, 값 벡터 만들기
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 2 단계: 어텐션 점수 ω 계산 - 단어끼리가 아니라, 쿼리와 키로 바꾼 벡터 사이에 닷곱
        attn_scores = queries @ keys.T  # omega
        # 3단계: 중요도 점수 α 계산 - 소프트맥스 처리...키 차원의 제곱근으로 스케일링
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 4단계: 컨텍스트 벡터를 만든다 - 이것도 α에 단어를 곱하는게 아니라 value 곱
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 버전 1과 다른 점은 여기서 Parameter 대신 Linear를 사용한다는 점...
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 1단계 쿼리, 키, 값 만들 때 행렬 곱이 아니라 순전파...
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
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

    # 셀프 어텐션 버전 1 사용
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    # 셀프 어텐션 버전 2 사용...
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
