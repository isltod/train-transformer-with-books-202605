import matplotlib.pyplot as plt
from ch5 import *

# 15 에포크에 초기 lr과 최대치
n_epochs = 15
initial_lr = 0.0001
peak_lr = 0.01
# 데이터 수에 에포크 수를 곱하면 스텝 수? 배치가 1인 경우인가...
total_steps = len(train_loader) * n_epochs
# 그 중에 20% 27스텝을 웜업으로 사용한다...
warmup_steps = int(0.2 * total_steps)  # 20% warmup
print(warmup_steps)
# 웜업 한 번에 올릴 lr
lr_increment = (peak_lr - initial_lr) / warmup_steps

global_step = -1
track_lrs = []

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        # 초기 20% 스텝 이내면 조금씩 lr 올리고
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            # 그 이후엔 최대치 사용...
            lr = peak_lr

        # 바꾼 lr을 옵티마이저의 param_groups에 적용시켜야 작동하는 모양...
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # param_groups에 적용한 lr은 다 같으니까 첫 번째만 기록했다 그래프로...
        track_lrs.append(optimizer.param_groups[0]["lr"])

        # Calculate loss and update weights
        # ...

plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")
plt.xlabel("Step")
total_training_steps = len(train_loader) * n_epochs
plt.plot(range(total_training_steps), track_lrs)
plt.tight_layout()
plt.show()
