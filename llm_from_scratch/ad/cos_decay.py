import matplotlib.pyplot as plt
from ch5 import *
import math

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

min_lr = 0.1 * initial_lr
track_lrs = []
global_step = -1

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        # 초기 20% 스텝 이내면 선형적으로 lr 올리고
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            # 그 이후엔 Cosine annealing
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            # cos0 -> 1 ~ cosπ -> 0 으로 감쇠
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # 바꾼 lr을 옵티마이저의 param_groups에 적용시켜야 작동하는 모양...
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])

        # Calculate loss and update weights

plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")
plt.xlabel("Step")
plt.plot(range(total_steps), track_lrs)
plt.tight_layout()
plt.show()
