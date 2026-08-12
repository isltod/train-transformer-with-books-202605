import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import time
from accelerate import Accelerator
from calc_loss import calc_loss_batch, calc_loss_loader, load_dataset_and_loaders
from gen_text_simple import text_to_token_ids, token_ids_to_text, GPT_CONFIG_124M
from previous_chapters import GPTModel, generate_text_simple
from pathlib import Path


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # 원본 코드 그대로 돌리면 시스템 멈춰서 guide-10-project 참고해서 Accelerator 도입...
    device = accelerator.device
    # 손실과 지금까지 처리한 토큰 수를 추적하기 위해 리스트를 초기화합니다.
    train_losses, val_losses, track_tokens_seen = [], [], []
    # 학습에 사용된 토큰 수, global_step은 -1에서 시작?
    tokens_seen, global_step = 0, -1

    # 메인 훈련 루프를 시작합니다.
    for epoch in range(num_epochs):
        model.train()  # 모델을 훈련 모드로 설정합니다.

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 이전 배치 반복에서 얻은 손실의 그레이디언트를 초기화합니다.
            # 크로스 엔트로피 loss 구하고
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            accelerator.backward(loss)  # 손실의 그레이디언트를 계산합니다.
            optimizer.step()  # 손실의 그레이디언트를 사용하여 모델 가중치를 업데이트합니다.
            tokens_seen += input_batch.numel()
            global_step += 1

            # 추가적인 평가 단계 - 전체 훈련 데이터와 검증 데이터에 대해서 평균 손실을 계산한다...
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"에포크 {epoch+1} (Step {global_step:06d}): "
                    f"훈련 손실 {train_loss:.3f}, 검증 손실 {val_loss:.3f}"
                )

        # 각 에포크 후 예제 문장 뒤에 모델이 예측한 문장을 출력...
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 모델은 평가모드 - 드롭아웃, 배치 정규화 등 끄고
    model.eval()
    # 역전파 차단...
    with torch.no_grad():
        # 훈련 데이터 전체로 평균 손실 계산
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        # 검증 데이터로 평균 손실 계산
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # 모델은 평가모드...
    model.eval()
    # 문장 길이 - 모델의 위치 임베딩은 문장 내 단어들에 꼬리표니까 (단어순서, 단어표현) 차원
    context_size = model.pos_emb.weight.shape[0]
    # 시작 문장을 단어 ID 목록으로...
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 시작 문장 짧게 주고, 한 번에 한 단어씩 예측해서 이어붙인 결과 받기...
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    # 시작 문장 + 예측 문장의 단어 ID 목록을 다시 문장으로...
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # 간결한 출력 포맷을 위해
    # 돌아갈 때는 모델을 다시 훈련 모드로 돌리고 보낸다...
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 에포크에 대한 훈련 손실과 검증 손실의 그래프를 그립니다.
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    # x축 라벨을 정수만 표시한다고...
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 처리한 토큰 수에 대한 두 번째 x 축을 만드는데 y 축을 공유하는 두 번째 x 축으로 만든다...
    # 두 번째는 위쪽에 표시된다...
    ax2 = ax1.twiny()
    # 사실 처리 토큰 수 대비 훈련 손실 그래프는 표시하지 않고, 단지 눈금 정렬을 위해 사용하므로 투명하게 alpha=0
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Accelerator 1. 기본 설정...허깅페이스 라이브러리...
    # 파이토치 학습코드를 멀티 GPU, TPU, 혼합정밀도 FP/BF 16 환경에서 실행할 때 복잡한 보일러플레이트 코드를 줄여주는 도구라...
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    if device.type == "cuda":
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")

    start_time = time.time()

    # 데이터 로더 및 토크나이저 준비
    _, train_loader, val_loader, tokenizer = load_dataset_and_loaders()

    # 모델 초기화
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Accelerator 2. 모델, 옵티마이저, 데이터 로더 감싸기...
    # Accelerator로 model, optimizer, dataloaders 준비 (디바이스 이동 포함)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        accelerator,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"훈련 소요 시간: {execution_time_minutes:.2f}분.")

    # 훈련 및 검증 손실 변화 시각화
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 모델과 옵티마이저를 저장
    # 간단하게 가중치만 저장하려면 torch.save(model.state_dict(), "model.pth")
    save_dir = Path(__file__).resolve().parents[2] / "data"
    save_path = save_dir / "toy_gpt2_params_and_optimizer.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
