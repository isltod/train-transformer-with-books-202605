from importlib.metadata import version
from gpt_download import download_and_load_gpt2
from gen_text_simple import (
    GPT_CONFIG_124M,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
)
from temp_scaling import generate
from previous_chapters import GPTModel
import numpy as np
import torch

print("텐서플로 버전:", version("tensorflow"))
print("tqdm 버전:", version("tqdm"))

# GPT2 가중치 다운로드 - 1억 2400만...
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("설정:", settings)
print("파라미터 딕셔너리 키:", params.keys())
print(params["wte"])
print("토큰 임베딩 가중치 텐서의 차원:", params["wte"].shape)

# 딕셔너리로 GPT2 설정을 만들어두고...
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 기본 설정을 특정 값으로 업데이트합니다.
model_name = "gpt2-small (124M)"  # 모델 이름
# 값이나 키는 별도지만 내부 리스트나 딕셔너리는 공유되는 얕은 복사...
NEW_CONFIG = GPT_CONFIG_124M.copy()
# model_configs 사전으로 NEW_CONFIG 사전 업데이트
NEW_CONFIG.update(model_configs[model_name])
# 연습을 위해 바꿨던 설정값 원래대로...
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 이렇게 하면 원래 GPT2와 같은 구조의 모델이 되는 모양...
gpt = GPTModel(NEW_CONFIG)
gpt.eval()


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


load_weights_into_gpt(gpt, params)

# 모델을 gpu로...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("현재 GPU 번호:", torch.cuda.current_device())
    print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")
    device = torch.device("cpu")

gpt.to(device)

torch.manual_seed(123)

# 테스트 문장 생성...
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))
