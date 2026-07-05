import torch
from torch import nn
from batch_casual import batch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 여기 d_out은 개별 dout x 헤드 수이므로 당연히 나눠 떨어져야 한다...
        # 일단 d_out = 2, 각 맥락 벡터 차원 d_out은 1
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어 떨어져야 합니다"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # 원하는 출력 차원에 맞도록 투영 차원을 낮춥니다.

        # 쿼리, 키, 값 가중치는 앞과 같고...
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 마지막에 개별 맥락 행렬 붙인걸 여기다 넣는데...꼭 필요한 건 아니지만 많은 LLM이 이걸 쓴다고...
        self.out_proj = nn.Linear(d_out, d_out)
        # 드롭아웃과 마스크는 앞과 같고...
        self.dropout = nn.Dropout(dropout)
        # 여기서 context_length는 단어 수 6을 그대로 쓰고...
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # 배치, 단어 수, 임베딩 차원
        b, num_tokens, d_in = x.shape
        # batch_causal에서 봐서 문제가 되는지는 알겠는데...왜 이런 경우가 생기는지...
        # `CausalAttention`과 마찬가지로, 입력의 `num_tokens`가 `context_length`를 넘는 경우 마스크 생성에서 오류가 발생합니다.
        # 실제로는 forward 메서드에 들어오기 전에 LLM이 입력이 `context_length`를
        # 넘지 않는지 확인하기 때문에 문제가 되지 않습니다.

        # 키, 쿼리, 값 행렬을 만드는 건 앞과 같고...
        keys = self.W_key(x)  # 크기: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # `num_heads` 차원을 추가함으로써 암묵적으로 행렬을 분할합니다. 위에서 head_dim = d_out // num_heads
        # 그다음 마지막 차원을 `num_heads`에 맞춰 채웁니다: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 2개 헤드에 d_out = 2이니까, 개별 맥락 벡터는 1차원이다...(2,6,2) -> (2,6,2,1)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 결국 중요한 건 단어 수, 단어 표현 차원이므로, 텐서 곱에서 이 둘이 맨 뒤로 가도록 전치...
        # 전치: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # (2,6,2,1) -> (2,2,6,1)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 코잘 마스크로 스케일드 점곱 어텐션(셀프 어텐션)을 계산합니다.
        # (2,2,6,1) x (2,2,1,6) = (2,2,6,6)...텐서 곱은 항상 끝에 두 차원을 맞춰주는 모양...
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # 각 헤드에 대해 점곱을 수행합니다.

        # 마스크를 불리언 타입으로 만들고 토큰 개수로 자르고, 상삼각을 -inf로 채우는건 같고...
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 키의 마지막 차원의 제곱근으로 스케일링하고 드롭아웃 적용도 같고...
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 개별 맥락 행렬을 이어 붙이려면 헤드 수와 헤드 차원이 붙어야 하니 다시 전치...
        # (b 2, num_tokens 6, num_heads 2, head_dim 1)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 그걸 이어 붙이는 것도 같은데... self.d_out = self.num_heads * self.head_dim
        # contiguous는 행렬의 좌상에서 우하로 가면서 값이 메모리에 순서대로 저장되어야 하는데,
        # view나 transpose 등을 해서 이 규칙이 깨져 있으면, 그걸 다시 좌상->우하 순으로 배열시키는 함수라고...
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 순전파 처리 전후 shape도 같은데...왜 이걸 쓰는지...
        context_vec = self.out_proj(context_vec)  # 투영

        return context_vec


if __name__ == "__main__":
    # 이것도 batch_causal에서...
    # torch.manual_seed(123)

    # 원래 (배치, 단어 수, 임베딩)인데...단어 수가 context_len, 임베딩 차원이 d_in
    batch_size, context_length, d_in = batch.shape
    # d_out은 2이지만 개별 d_out은 1, 그걸 헤드 수만큼 합친게 2가 된다...
    # 이걸 basic에서처럼 2 x 2 = 4로 바꾸면 basic과 결과가 같다...
    d_out = 2
    # 헤드도 하던대로 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    # 순전파로 맥락 벡터 행렬을 받는 것도 같고...
    context_vecs = mha(batch)

    print(context_vecs)
    # 그래서 여긴 basic (2,6,4)와 다르게 (2,6,2)
    print("context_vecs.shape:", context_vecs.shape)
