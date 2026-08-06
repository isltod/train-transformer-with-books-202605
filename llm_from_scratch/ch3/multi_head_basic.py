import torch
from torch import nn
from batch_casual import CausalAttention, batch


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # CausalAttention이 Module을 상속받고, 멀티 헤드라는 건 결국 Module의 리스트 같은거...
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # 순전파는 헤드마다 돌면서 순전파 시키기 정도...그걸 마지막 차원으로 이어붙이기...결과가 (2, 6, 2+2)
        # 근데 어차피 결과가 이어붙이기로 나온다면, x(6,3) @ w(3,2) -> (6,2)인데, w를 (3,4)로 하면 (6,4)가 나오잖아...
        return torch.cat([head(x) for head in self.heads], dim=-1)


if __name__ == "__main__":
    # 이건 import에서 물려받고
    # torch.manual_seed(123)

    # (배치, 단어 수, 임베딩)에서 단어 수가 맥락 수...
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
