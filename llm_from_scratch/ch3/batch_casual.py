import torch
import torch.nn as nn

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

torch.manual_seed(123)
# 배치 흉내를 위해 input을 행으로 쌓기...(배치 2, 단어 수 6, 임베딩 3)
batch = torch.stack((inputs, inputs), dim=0)


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        # self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # 추가
        # register_buffer는 일단 변수 선언인데, 모델이 gpu 이동하면 따라 이동하는 방식으로 선언한다고...
        # 상삼각 부분만 1로 채운 텐서를 mask 속성으로...
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # 추가

    def forward(self, x):
        # 배치, 단어 수, 임베딩 차원
        b, num_tokens, d_in = x.shape
        # 일단 키, 쿼리, 값 생성은 뻔하고...
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # transpose는 두 차원 맞교환...1번과 2번 차원 교환...(2,2,6)
        # 배치 제외하고 보면 (6,2) 행렬에 곱해야 하므로 (2,6)으로 전치한다...결과는 (2,6,6)
        attn_scores = queries @ keys.transpose(1, 2)
        # 마스킹 위치인 상삼각 부분을 -inf로 채우는데...
        # :num_tokens 슬라이싱으로 단어 수 까지만 사용하고 문맥이 길면 자르는데...이게 왜 다른 경우가 있는거지?
        # 여기서 mask는 (6,6)인데, 이상하게 단어 수 num_tokens가 6을 넘으면 문제...근데 이런 경우가 왜 있지?
        # 근데 실제로는 forward 메서드에 들어오기 전에 LLM이 입력이 `context_length`를 넘지 않는지 확인하기 때문에 문제가 없다고...
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 키의 임베딩 차원의 제곱근으로 스케일링...
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 추가

        # 중요도 점수 α에 values 곱해서 문맥 벡터 생성...
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    print(
        batch.shape
    )  # 각각 여섯 개의 토큰으로 구성된 두 개의 입력. 각 토큰의 임베딩 차원은 3입니다.

    # batch는 (배치, 단어 수, 임베딩)이므로 그 중 단어 수를 context_length로...
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)

    # 순전파로 맥락 벡터 구하고...
    context_vecs = ca(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
