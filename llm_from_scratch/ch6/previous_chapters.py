import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 텍스트를 토큰화합니다.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert (
            len(token_ids) > max_length
        ), "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."

        # 슬라이딩 윈도를 사용해 책을 max_length 길이의 중첩된 시퀀스로 나눕니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            # 토치 텐서로 리스트에 넣는다...
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):

    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")
    # 데이터셋을 만듭니다.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # 데이터 로더를 만듭니다.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 여기 d_out은 개별 dout x 헤드 수, 여기선 개별 맥락 벡터 dout이 1로 줄어서 d_out = 2
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어 떨어져야 합니다"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # 원하는 출력 차원에 맞도록 투영 차원을 낮춥니다.

        # 쿼리, 키, 값 가중치 마지막 차원은 num_heads 만큼 이어붙인 차원(1 x 2 = 2)
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
        keys = self.W_key(x)  # 크기: (b, num_tokens, d_out=개별out x num_heads)
        queries = self.W_query(x)
        values = self.W_value(x)

        # `num_heads` 차원을 추가함으로써 암묵적으로 행렬을 분할합니다. 위에서 head_dim = d_out // num_heads
        # 그다음 마지막 차원을 `num_heads`에 맞춰 채웁니다: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 2개 헤드에 d_out = 2이니까, 개별 맥락 벡터는 1차원이다...(2,6,2) -> (2,6,2,1)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 결국 어텐션 연산은 단어대 단어 연산이므로 연산될 차원 둘(단어순서, 단어표현)이 맨 뒤로 가도록 전치...
        # 전치: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # (2,6,2,1) -> (2,2,6,1)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 코잘 마스크로 스케일드 점곱 어텐션(셀프 어텐션)을 계산합니다.
        # (2,2,6,1) x (2,2,1,6) = (2,2,6,6)...단어대 단어 닷곱이므로 (단어순서, 단어표현) 전치해서 곱...
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # 각 헤드에 대해 점곱을 수행합니다.

        # 마스크를 불리언 타입으로 만들고 토큰 개수로 자르고, 상삼각을 -inf로 채우는건 같고...
        # 차원이 늘어났지만 어차피 2차원 행렬을 브로드캐스팅...
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


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 분모가 0이 안되게...
        self.eps = 1e-5
        # 정규화 결과를 늘리고 이동시키는 매개변수들인데, 모델 성능에 도움이 된다면 학습해서 변형시킨다...
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 정규화는 마지막 차원에 적용...
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False 또는 토치 2.0이상에서 correction=0은 데이터 수 N으로 나누는 모분산...반대는 N-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 분모가 0이 안되게...
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# 대략 -3 이하는 0에 아주 가깝고, 0까지는 음수가 나오고, 그 중 -0.75 정도에서 미분이 0이되는 함수...
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


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


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 책 156 쪽 구조
        # 근데 Linear 층은 W_T 구조로 저장하는데, Embedding 층은 그냥 W 구조로 저장하네...(50257, 768)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 이 드롭아웃은 트랜스포머 들어가기 전에 거치는 층
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 이게 핵심...트랜스포머 블록
        self.trf_blocks = nn.Sequential(
            # 이 블록이 모델 크기에 따라 여러번 반복...
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 학습 안정화와 관련 있다고..
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 마지막 완전연결은 편향 b 없이...여긴 W_T라서 (50257, 768)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 크기 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# 맨 처음 idx는 현재 문장이 담긴 (batch, n_tokens) 크기의 인덱스 배열로 시작
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 새로 생성할 토큰 숫자만큼 반복해서...
    for _ in range(max_new_tokens):

        # idx는 (배치, 각 문장의 토큰 ID들) 텐서...받은 문장들이 위치 임베딩 길이보다 크면 잘라내고...
        idx_cond = idx[:, -context_size:]

        # 역전파 끊고 모델 예측 받고
        with torch.no_grad():
            logits = model(idx_cond)

        # 모델 예측 결과는 (배치, 단어 순서, 그 순서 단어가 단어 사전에 있는 단어들에 가까운 정도) 텐서를 내는데...
        # 이 모델은 마지막 단어 1개씩 예측하고, 그걸 다시 입력으로 사용하니까 단어 순서 인덱스를 -1로 하나만 뽑는다...
        # 즉 (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 단어 사전 단어들 중 가장 확률 높은 단어 선택...이 방식은 greedy decoding이라고...
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 선택된 단어 ID를 (배치, 단어들) 목록에 추가하고 다음 단어 예측...
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


#####################################
# Chapter 5
####################################


# OpenAI GPT2 가중치를 myGPT 가중치에 할당하기...
# 근데 이걸 왜 load 같은 함수를 사용 못하고 하나하나 처리해야 하지? tensorflow 구조를 pytroch로 옮겨서 그런가?
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"크기가 다릅니다. left: {left.shape}, right: {right.shape}")
    # 마지막은 훈련 가능한 토치 텐서로 반환
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    # OpenAI 매개변수들을 내가 만든 모델이 넣는데...사전 키가 이렇다는 건 그냥 받아들여야...
    # 위치 임베딩과 토큰 임베딩 가중치...
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    # 토큰 임베딩 가중치는 그대로 마지막 출력 임베딩 가중치에도 사용...
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # 트랜스포머 블록의 트랜스포머마다 반복하는 모양이고...
    for b in range(len(params["blocks"])):
        # 어텐션 query, key, value 가중치
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        # 어텐션 편향
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        # 출력 가중치와 편향
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # 피드 포워드 가중치와 편향
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # 층 정규화 가중치와 편향...
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    # 최종 정규화, 출력...
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 마지막 출력층은 토큰 임베딩 가중치 그대로 사용
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
