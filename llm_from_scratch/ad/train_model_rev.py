import matplotlib.pyplot as plt
import tiktoken
import time
import math
from accelerate import Accelerator
from previous_chapters import (
    evaluate_model,
    generate_and_print_sample,
    calc_loss_batch,
    plot_losses,
)
from ch5 import *

ORIG_BOOK_VERSION = False


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    n_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    warmup_steps,
    initial_lr=3e-05,
    min_lr=1e-6,
):

    device = accelerator.device
    if device.type == "cuda":
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")

    # 손실, 지금까지 처리한 토큰 수, 학습률 추적
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    # global_step은 에포크 무시하고 총 스텝 수...
    tokens_seen, global_step = 0, -1

    # peak_lr, total_training_steps는 이미 밖에서 계산 또는 설정 했는데...그걸 전달 아니고 계산하게 처리하네...
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs

    # warmup 기간동안 학습률 증가분...
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    # 일단 학습 루프는 에포크 단위로 도는게 맞고...
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 초기 20% 스텝 이내면 선형적으로 lr 올리고
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                # 그 이후엔 Cosine annealing
                progress = (global_step - warmup_steps) / (
                    total_training_steps - warmup_steps
                )
                # cos0 -> 1 ~ cosπ -> 0 으로 감쇠
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            # 바꾼 lr을 옵티마이저의 param_groups에 적용시켜야 하고...
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # 크로스 엔트로피 손실 계산하고 역전파...
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 3. 그냥 loss의 역전파가 아니라 accelerator의 역전파 호출...
            # loss.backward()
            accelerator.backward(loss)

            # 놈을 1로 줘서 gradient clipping
            if ORIG_BOOK_VERSION:
                # 이건 뭐 원래 책이 문제가 있다면서 고치는 부분인 듯...중요한 건 > 냐 >= 문제인 듯...
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 가중치 갱신하고 처리한 토큰 수 누적하고...
            optimizer.step()
            tokens_seen += input_batch.numel()

            # 주기적으로 추가적 평가 - 전체 훈련 데이터와 검증 데이터에 대해서 평균 손실을 계산한다...
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 현재의 훈련/검증 손실 보고...
                print(
                    f"Ep {epoch+1} (Iter {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # 각 에포크 후 예제 문장 뒤에 모델이 예측한 문장을 출력...
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs


# Accelerator 1. 기본 설정...허깅페이스 라이브러리...
accelerator = Accelerator(mixed_precision="bf16")

peak_lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

# Accelerator 2. 모델, 옵티마이저, 데이터 로더 감싸기...
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

tokenizer = tiktoken.get_encoding("gpt2")

n_epochs = 15
# 데이터 수에 에포크 수를 곱하면 스텝 수? 배치가 1인 경우인가...
total_steps = len(train_loader) * n_epochs
# 그 중에 20% 27스텝을 웜업으로 사용한다...
warmup_steps = int(0.2 * total_steps)  # 20% warmup

# 실행 시간 측정
start_time = time.time()

train_losses, val_losses, tokens_seen, lrs = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    n_epochs=n_epochs,
    eval_freq=5,
    eval_iter=1,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
    warmup_steps=warmup_steps,
    initial_lr=1e-5,
    min_lr=1e-5,
)

# 실행 시간 측정
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 학습률 변화 그래프
plt.figure(figsize=(5, 3))
plt.plot(range(len(lrs)), lrs)
plt.ylabel("Learning rate")
plt.xlabel("Steps")

# 손실 변화 그래프...
epochs_tensor = torch.linspace(1, n_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
plt.tight_layout()
plt.show()
