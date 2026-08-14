import tiktoken
import torch
from previous_chapters import GPTModel
from gpt_download import download_and_load_gpt2


def classify_review(
    text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.eval()

    # 모델에 대한 입력 준비
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # 너무 긴 시퀀스 자르기
    input_ids = input_ids[: min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length가 지정되지 않았습니다. 모델의 최대 문맥 길이를 사용하려면"
        "max_length=model.pos_emb.weight.shape[0]로 지정하세요."
    )
    assert (
        max_length <= supported_context_length
    ), f"max_length({max_length})가 모델이 지원하는 문맥 길이({supported_context_length})를 초과했습니다."
    # 또는 max_length=None인 경우를 안정적으로 처리하는 방법은 다음과 같습니다.
    # max_len = min(max_length, supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]

    # 가장 긴 시퀀스로 패딩하기
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 배치 차원 추가

    # 모델 추론
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # 마지막 출력 토큰의 로짓
    predicted_label = torch.argmax(logits, dim=-1).item()

    # 분류 결과 반환
    return "스팸" if predicted_label == 1 else "스팸아님"


def init_model(state_dict, device):
    # GPT-2 모델 설정
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)

    # 분류 헤드 교체 및 파라미터 동결/해제 설정
    # 1. 일단 다 동결하고
    for param in model.parameters():
        param.requires_grad = False

    # 2. 2개로 분류하는 마지막 헤드는 교체...하면 바로 학습 모드로...
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    # 3. 마지막 트랜스포머 블록과 층 정규화 층은 성능을 높이려고 학습 모드로 한다고...
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 그리고 이전에 학습했던 가중치들 넣고 모델 반환
    model_state_dict = torch.load(state_dict, map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    TRAIN_DS_MAX_LEN = 120
    model = init_model("review_classifier.pth", device)

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(
        classify_review(text_1, model, tokenizer, device, max_length=TRAIN_DS_MAX_LEN)
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(
        classify_review(text_2, model, tokenizer, device, max_length=TRAIN_DS_MAX_LEN)
    )

"""
이런식으로 사용할 수 있다...여기서 얻은 교훈을 다시 한 번 생각해보면...
1. 그냥 다음 시퀀스를 예측하라고 학습시켰는데, 이거 스팸이야? 질문에 나름 대답(비록 엉망이지만)을 하는 창발성이 있었다.
2. 모델 구조상 예측 결과 텐서의 마지막 벡터만 유효한 로짓이었다.
3. 단지 마지막 출력 로짓의 차원만 변경했는데(+fine tuning을 하긴 했다...), 스팸 분류를 한다.
전반적으로는...언어라는 도메인에서 이뤄지는 일은 사실 다음 단어를 꾸며내는 능력의 변이 정도가 아닐까...
그렇다면 뭔가 하려면 그 도메인의 핵심 능력을 찾아야 하는게 아닐까?
또는 사람이 하는 일이란 도 다음 단어를 꾸며내는 능력의 변이 정도는 아닐까? 하는 생각이 든다...
"""
