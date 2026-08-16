import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from data_loader import val_data
from get_data import format_input

if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 50257,  # 어휘사전 크기
        "context_length": 1024,  # 문맥 길이
        "drop_rate": 0.0,  # 드롭아웃 비율
        "qkv_bias": True,  # 쿼리-키-값 편향
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)

    # 일단 능동->수동 지시어 하나 넣어보는데...
    input_text = format_input(val_data[0])
    print(input_text)

    tokenizer = tiktoken.get_encoding("gpt2")
    # 답을 생성하고...
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    # 문장으로 바꾸고
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 질문 부분은 건너뛰고 답만 출력해보면...
    response_text = (
        generated_text[len(input_text) :].replace("### Response:", "").strip()
    )
    # 당연하지만, 훈련이 전혀 안되어 있으니 능동 수동 전환이 아니라 그냥 단순 반복...
    print(response_text)
