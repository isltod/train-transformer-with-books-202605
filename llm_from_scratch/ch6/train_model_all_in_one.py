# 뭔가 꼬여서 from 문을 하나로 모아서 처리한다...
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator

# 이전 장의 GPT 모델 및 가중치 로더 가져오기
from previous_chapters import GPTModel, load_weights_into_gpt
from gpt_download import download_and_load_gpt2


# ==========================================
# 1. Dataset & DataLoader 준비
# ==========================================
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# ==========================================
# 2. 평가 및 손실 계산 함수
# ==========================================
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # 마지막 토큰의 로짓
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples if num_examples > 0 else 0.0


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 마지막 토큰의 로짓
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ==========================================
# 3. 훈련 루프 함수
# ==========================================
def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    num_epochs,
    eval_freq,
    eval_iter,
):
    # 이 모델의 경우 CPU와 GPU는 말도 안되게 차이가 난다...LLM은 꼭 GPU를 써야 하는 걸까?
    device = accelerator.device
    print(f"사용 장치: {device}")
    if device.type == "cuda":
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            global_step += 1
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            accelerator.backward(loss)
            optimizer.step()
            examples_seen += input_batch.shape[0]

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"에포크 {epoch+1} (Step {global_step:06d}): "
                    f"훈련 손실 {train_loss:.3f}, 검증 손실 {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(
            f"에포크 {epoch+1} 완료 -> 훈련 정확도: {train_accuracy*100:.2f}% | 검증 정확도: {val_accuracy*100:.2f}%"
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.show()


# ==========================================
# 4. 메인 실행 블록
# ==========================================
def main():
    # Accelerator 설정 (혼합 정밀도)
    accelerator = Accelerator(mixed_precision="bf16")

    # 1) 토크나이저 및 데이터셋/데이터로더 생성
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(
        csv_file="train.csv", max_length=None, tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    batch_size = 8
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, drop_last=False
    )

    # 2) GPT-2 모델 로드 및 설정
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

    torch.manual_seed(123)
    # GPT 모델을 사용한다면, 여기 두 가지 함수가 일단 중요...
    # 1. GPT 모델의 구조를 알아야 여기서 같은 구조를 만들고
    model = GPTModel(BASE_CONFIG)
    # 2. 사전 훈련 가중치 구조를 알아야 여기서 가중치를 로드시킬 수 있다..
    load_weights_into_gpt(model, params)

    # 3) 분류 헤드 교체 및 파라미터 동결/해제 설정
    # 1. 일단 다 동결하고
    for param in model.parameters():
        param.requires_grad = False

    # 2. 2개로 분류하는 마지막 헤드는 교체...하면 바로 학습 모드로...
    num_classes = 2
    model.out_head = nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    # 3. 마지막 트랜스포머 블록과 층 정규화 층은 성능을 높이려고 학습 모드로 한다고...
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 4) 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # 5) Accelerator에 모델, 옵티마이저, 데이터로더 등록
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 6) 훈련 시작
    num_epochs = 5
    start_time = time.time()

    print("\n--- 훈련 시작 ---")
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"\n훈련 소요 시간: {execution_time_minutes:.2f}분")

    # 7) 최종 결과 평가
    # 요것도 간과했는데, 최종 테스트 로더도 Accelerator로 감싸야 한다...
    test_loader_prepared = accelerator.prepare(test_loader)
    train_accuracy = calc_accuracy_loader(train_loader, model, accelerator.device)
    val_accuracy = calc_accuracy_loader(val_loader, model, accelerator.device)
    test_accuracy = calc_accuracy_loader(
        test_loader_prepared, model, accelerator.device
    )

    print(f"\n--- 최종 평가 결과 ---")
    print(f"훈련 정확도: {train_accuracy*100:.2f}%")
    print(f"검증 정확도: {val_accuracy*100:.2f}%")
    print(f"테스트 정확도: {test_accuracy*100:.2f}%")

    # 8) 모델 저장
    torch.save(model.state_dict(), "review_classifier.pth")

    # 9) 시각화
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss"
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy"
    )


if __name__ == "__main__":
    main()
