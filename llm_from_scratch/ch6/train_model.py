from huggingface_hub.utils import tqdm
from lose_n_accuracy import calc_loss_batch, calc_loss_loader, calc_accuracy_loader
from add_cls_head import model
from make_datatloader import train_loader, val_loader, test_loader
from accelerate import Accelerator
import torch
import time
import matplotlib.pyplot as plt


# 5장과 동일
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# 5장의 `train_model_simple`과 전체적으로 동일
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
    device = accelerator.device
    if device.type == "cuda":
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")

    # 훈련/검증 손실과 정확도, 지금까지 처리한 토큰 수 추적
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # global_step은 에포크 무시하고 총 스텝 수...
    examples_seen, global_step = 0, -1

    # 메인 학습 루프
    for epoch in tqdm(range(num_epochs)):
        model.train()  # 모델을 훈련 모드로 설정

        for input_batch, target_batch in train_loader:
            global_step += 1
            # 이전 배치 반복에서 얻은 손실의 그레이디언트 재설정
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 3. 그냥 loss의 역전파가 아니라 accelerator의 역전파 호출...
            # loss.backward()
            accelerator.backward(loss)
            # 가중치 갱신하고 처리한 토큰 수 누적하고...
            optimizer.step()
            examples_seen += input_batch.shape[0]

            # 주기적으로 추가적 평가 - 전체 훈련 데이터와 검증 데이터에 대해서 평균 손실을 계산한다...
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

        # 각 에포크 후 훈련/검증 정확도 계산
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"훈련 정확도: {train_accuracy*100:.2f}% | ", end="")
        print(f"검증 정확도: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# Accelerator 1. 기본 설정...허깅페이스 라이브러리...
accelerator = Accelerator(mixed_precision="bf16")

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# Accelerator 2. 모델, 옵티마이저, 데이터 로더 감싸기...
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"훈련 소요 시간: {execution_time_minutes:.2f}분")


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 훈련 및 검증 손실을 에포크에 따라 그립니다.
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 처리한 샘플 수를 위해 두 번째 x축을 만듭니다.
    ax2 = ax1.twiny()  # 동일한 y축을 공유하는 두 번째 x축을 만듭니다.
    # 눈금 정렬을 위한 보이지 않는 그래프
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # 공간을 확보하기 위해 레이아웃을 조정합니다.
    plt.show()


# 훈련/검증 손실 그래프...
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
# 훈련/검증 정확도 그래프
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

train_accuracy = calc_accuracy_loader(train_loader, model, accelerator.device)
val_accuracy = calc_accuracy_loader(val_loader, model, accelerator.device)
test_accuracy = calc_accuracy_loader(test_loader, model, accelerator.device)

print(f"훈련 정확도: {train_accuracy*100:.2f}%")
print(f"검증 정확도: {val_accuracy*100:.2f}%")
print(f"테스트 정확도: {test_accuracy*100:.2f}%")

# 뭔가 꼬여서 GPU를 잡질 못한다...from에서 참조하고 있는 파일들 정리해야 한다...
