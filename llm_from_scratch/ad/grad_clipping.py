from ch5 import *
from previous_chapters import calc_loss_batch
from accelerate import Accelerator

# Accelerator 1. 기본 설정...허깅페이스 라이브러리...
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

# Accelerator 2. 모델, 옵티마이저, 데이터 로더 감싸기...
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

input_batch, target_batch = next(iter(train_loader))
# 이 부분 때문에 그냥 실행하면 시스템이 죽는다...그래서 Accelerator 도입해야 실행된다...
loss = calc_loss_batch(input_batch, target_batch, model, device)
# 역전파로 기울기를 grad에 저장...
loss.backward()


def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


print(find_highest_gradient(model))

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(find_highest_gradient(model))
