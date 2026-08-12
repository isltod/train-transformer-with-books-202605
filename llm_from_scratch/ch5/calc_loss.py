import os
import torch
import tiktoken
from gen_text_simple import GPT_CONFIG_124M
from previous_chapters import create_dataloader_v1, GPTModel
from pathlib import Path

# 파일 경로를 상대경로/절대경로 호환성 있게 설정
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = SCRIPT_DIR / "the-verdict.txt"


def load_dataset_and_loaders(file_path=DEFAULT_FILE_PATH, batch_size=2):
    tokenizer = tiktoken.get_encoding("gpt2")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    return text_data, train_loader, val_loader, tokenizer


def calc_loss_batch(input_batch, target_batch, model, device):
    # train_model에서 호출될 때는 Accelerator로 이미 gpu tensor라 이 코드가 필요 없는데,
    # 그냥 한 번 더 해도 문제는 없는듯...
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 순전파로 로짓(소프트맥스가 아니다...) 예측하고
    logits = model(input_batch)
    # 펼친 로짓과 펼친 정답지 ID로 크로스 엔트로피
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    # 데이터 로더에 데이터 없으면 nan 반환하고 종료...
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # 배치 수 없으면 한번에 배치 하나씩 그냥 모든 데이터를 처리...
        num_batches = len(data_loader)
    else:
        # num_batches가 데이터 로더에 있는 배치 개수보다 크면
        # num_batches를 데이터 로더에 있는 총 배치 개수로 맞춥니다.
        num_batches = min(num_batches, len(data_loader))

    # 그러니까, 원래 데이터로더만 돌려도 배치들이 반환되는데, enumerate를 하면 앞에 인덱스를 붙여준다...
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 배치 수 이내에서...
        if i < num_batches:
            # 배치의 크로스 엔트로피 손실을 구하고 누적...
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 평균 크로스 엔트로피 손실을 반환
    return total_loss / num_batches


if __name__ == "__main__":
    text_data, train_loader, val_loader, tokenizer = load_dataset_and_loaders()

    # 처음 99개 문자
    print(text_data[:99])
    # 마지막 99개 문자
    print(text_data[-99:])

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("문자:", total_characters)
    print("토큰:", total_tokens)

    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("훈련 토큰 수:", train_tokens)
    print("검증 토큰 수:", val_tokens)
    print("모든 토큰 수:", train_tokens + val_tokens)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")
        device = torch.device("cpu")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("훈련 손실:", train_loss)
    print("검증 손실:", val_loss)
