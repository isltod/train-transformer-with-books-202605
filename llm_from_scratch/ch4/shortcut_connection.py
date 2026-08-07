import torch
import torch.nn as nn
from gelu import GELU


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        # 숏컷을 사용할지 여부
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                # layer_sizes는 마지막 5만 1, 나머지는 다 3
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # 현재 층의 출력을 계산합니다.
            layer_output = layer(x)
            # 숏컷 연결 옵션이 True고, 숏컷이 입력을 출력에 더하므로 둘의 shape이 같다면 숏컷
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # 정방향 계산과 연습용 정답...둘 다 (1,1)
    output = model(x)
    target = torch.tensor([[0.0]])

    # 타깃과 출력의 가까운 정도를 기반으로 손실을 계산합니다.
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 그레이디언트를 계산하기 위한 역전파 - 연습용으로 한 번만 사용하므로 grad 0나 step 없음...
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            # 가중치의 그레이디언트의 평균 절댓값을 출력합니다.
            print(
                f"{name}의 평균 그레이디언트는 {param.grad.abs().mean().item()}입니다."
            )


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
# 망을 만들 때 가중치 랜덤 초기화가 있으므로 seed를 준다면 그 전에...
torch.manual_seed(123)
# 숏컷 없는 버전의 기울기
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print("숏컷 없는 버전의 기울기---------")
print_gradients(model_without_shortcut, sample_input)
# 같은 난수 사용하도록 시드를 다시 초기화...
torch.manual_seed(123)
# 숏컷 있는 버전의 기울기
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print("숏컷 있는 버전의 기울기---------")
print_gradients(model_with_shortcut, sample_input)
