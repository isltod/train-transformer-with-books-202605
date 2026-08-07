from dummy_gpt import GPT_CONFIG_124M
from feed_forward import FeedForward
from layer_norm import LayerNorm
from multi_head import MultiHeadAttention
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 책 152쪽 구조...
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 어텐션 블록을 위한 숏컷 연결
        shortcut = x
        # 층 정규화 1
        x = self.norm1(x)
        # 멀티헤드 셀프 어텐션
        x = self.att(x)  # 크기: [batch_size, num_tokens, emb_size]
        # 드롭아웃
        x = self.drop_shortcut(x)
        # 숏컷
        x = x + shortcut  # 원래 입력을 더합니다.

        # 피드 포워드 블록을 위한 숏컷 연결
        shortcut = x
        # 층 정규화 2
        x = self.norm2(x)
        # feedforward
        x = self.ff(x)
        # 드롭아웃
        x = self.drop_shortcut(x)
        # 숏컷
        x = x + shortcut  # 원래 입력을 더합니다.

        return x


torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # 크기: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
# shape은 같지만, 입력은 단어표현이고 출력은 문맥 벡터...
print("입력 크기:", x.shape)
print("출력 크기:", output.shape)
