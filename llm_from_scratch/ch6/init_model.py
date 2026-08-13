from previous_chapters import generate_text_simple, text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt
from make_datatloader import train_dataset, tokenizer

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

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

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"데이터셋 길이 {train_dataset.max_length}가 모델의 문맥 "
    f"길이 {BASE_CONFIG['context_length']}를 초과합니다. `max_length={BASE_CONFIG['context_length']}`로 "
    f"데이터 셋을 다시 초기화하십시오."
)

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

if __name__ == "__main__":

    # 모델이 제대로 로드되었는지 샘플 텍스트 넣고 문장 생성 테스트...
    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer))

    # fine tunning 전에 분류 능력이 있나 확인한다고? 그냥 이렇게 막 질문을 넣으면 답이 나오나?
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"],
    )
    # 당연히 엉뚱한 말을 뱉어내는데....
    print(token_ids_to_text(token_ids, tokenizer))
