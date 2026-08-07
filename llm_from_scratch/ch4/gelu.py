import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 대략 -3 이하는 0에 아주 가깝고, 0까지는 음수가 나오고, 그 중 -0.75 정도에서 미분이 0이되는 함수...
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


if __name__ == "__main__":
    # ReLU는 nn에서 받는데 GELU는 직접 만들어야 하나?
    gelu, relu = GELU(), nn.ReLU()

    # 샘플 데이터
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
