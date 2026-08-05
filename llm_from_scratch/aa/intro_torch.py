import numpy as np
import torch

"""
A.1. 버전, GPU 확인
"""
print(torch.__version__)
print(torch.cuda.is_available())

"""
A.2.1 텐서 랭크, 값 복사와 참조
"""
# 그냥 숫자를 쓰면 0D tensor (scalar)
tensor0d = torch.tensor(1)

# 대괄호 하나면 1D tensor (vector)
tensor1d = torch.tensor([1, 2, 3])

# 대괄호 2이면 2D tensor
tensor2d = torch.tensor([[1, 2], [3, 4]])

# 셋이면 3D tensor
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 또는 넘파이 배열에서...
ary3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tensor3d_val = torch.tensor(ary3d)  # 값을 복사해서 새로 만들기
tensor3d_ref = torch.from_numpy(ary3d)  # 넘파이 배열과 메모리를 공유?

# 확인을 위해서 원본의 맨 처음 요소 값을 바꿔보면...
ary3d[0, 0, 0] = 999
print(tensor3d_val)  # 값을 복사한 경우는 상관없고
print(tensor3d_ref)  # 참조한 경우는 값이 바뀌어있고...

"""
A.2.2 데이터 타입
"""
# 정수는 기본적으로 int64
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)
# 실수는 기본적으로 float32
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
# int64 -> float32 변환
floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)

"""
A.2.3 일반적인 연산자
"""
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d)

# 같은 결과 size(2,3) 출력
print(tensor2d.shape)
print(tensor2d.size())

# 결과는 같지만, view는 그냥 보기만 바꾸니 값들이 붙어있어야 하고,
# reshape는 복사해서 다시 만들어 값이 붙어있지 않아도 된다고...
aa = tensor2d.view(3, 2)
bb = tensor2d.reshape(3, 2)
print(aa)
print(bb)
# 근데 원본을 바꾸면 둘 다 바뀌어있기는 마찬가지다...
tensor2d[0, 0] = 999
print(aa)
print(bb)

print(tensor2d.T)
# 행렬곱은 메서드 matmul 또는 @ 연산자
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)

"""
A.3 계산그래프
"""
import torch.nn.functional as F

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1])  # input feature
# 학습 매개변수는 뒤에서 grad() 적용시키려면 requires_grad=True 있어야 하는모양...
w1 = torch.tensor([2.2], requires_grad=True)  # weight parameter
b = torch.tensor([0.0], requires_grad=True)  # bias unit

z = x1 * w1 + b  # net input
a = torch.sigmoid(z)  # activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)
# 여기까지 순전파, 계산그래프를 만든다...
"""
A.4 자동미분
"""
# 여기부터 역전파
from torch.autograd import grad

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)
print(grad_L_w1)
print(grad_L_b)

# 또는 더 고수준 backward() 함수 사용...근데 위에서 grad 하고 다시 backward 했는데 .grad 속성이 누적되지 않네?
loss.backward()
print(w1.grad)
print(b.grad)

"""
A.5 다층 신경망
"""


# nn.Module을 상속받는 클래스
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # 순서대로 연결해서 layers(x)으로 순전파하려고 Sequential 사용
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


torch.manual_seed(123)
model = NeuralNetwork(50, 3)

# 모델 구조 출력
print(model)

# 학습 매개변수면 갯수를 다 더해라...
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

# 첫 번째 레이어 w 매개변수를 출력...
# (50,30)으로 들어갔는데 (30,50)이 나오네...50 받아서 30으로 만들어라인데...Wx 순으로 곱한다는 얘기지...
# 그리고 마지막에 requires_grad=True...
print(model.layers[0].weight)
print(model.layers[0].weight.shape)

# 0~1 균등분포에서 (1, 50) 난수 텐서 만들기
# 이거 헛갈리는데? 열벡터면 (50,1) 아닌가? 행벡터로 만들어서 넣는데...곱할 때 바뀌나?
# 결과와 맞춰보면 실제 연산은 X @ w.T 모양인거 같은데...실제로도 그렇다네...
X = torch.rand((1, 50))
print(X.shape)
print(bb.shape)

# 순전파
out = model(X)
# grad_fn=<AddmmBackward0 - Addmm(행렬 곱 이후 덧셈) 연산에 대한 역전파 기울기이다...
print(out)
print(out.shape)

# 역전파 없이 순전파만 사용(테스트나 실제 예측 등 학습 필요없는 경우)
with torch.no_grad():
    out = model(X)
print(out)

# 또는 실제 필요한 건 확률이지만 학습은 로짓까지만 시킨다면...
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)

"""
A.6 데이터셋과 데이터로더
"""

X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)
y_test = torch.tensor([0, 1])

from torch.utils.data import Dataset


class ToyDataset(Dataset):
    # 데이터셋은 init, getitem, len 세 가지는 기본적으로 오버라이딩 해야하는 모양...
    # shape은 필요 없는 모양...
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


# 데이터 넣어서 데이터셋 만들면 데이터셋은 끝...별거 없고...
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
print(len(train_ds))

from torch.utils.data import DataLoader

torch.manual_seed(123)
# 데이터로더는 데이터셋으로 만들고, 배치, 셔플, 워커 지정이 끝...
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

# drop_last는 미니배치로 나눠 떨어지지 않는 마지막 부분 버리기...
train_loader = DataLoader(
    dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

""" A.7 전형적인 훈련 과정 """

import torch.nn.functional as F

# 1. 모델 네트워크와 옵티마이저 설정
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 2. 에포크 별로 돌면서 학습
num_epochs = 3
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        # 2.1 순전파
        logits = model(features)
        # 2.2 손실 계산
        loss = F.cross_entropy(logits, labels)  # Loss function
        # 2.3 기울기 비우고
        optimizer.zero_grad()
        # 2.4 역전파
        loss.backward()
        # 2.5 매개변수 수정
        optimizer.step()

        # 2.6 진행 보고
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

    # 2.7 기타 성능 테스트 등 별도 작업(학습 끄고 진행)
    model.eval()

# 3. 샘플 예측으로 성능 확인
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# 텐서 화면 출력 조절 - sci_mode=False는 지수 표시 안함
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)
predictions = torch.argmax(probas, dim=1)
print(predictions)

# 4. 정답과 예측 비교로 성능 평가
predictions == y_train
torch.sum(predictions == y_train)


# 또는 이렇게 평가 함수로...
def compute_accuracy(model, dataloader):

    # 평가 모드 학습 안함
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        # 기울기 계산은 그래도 하나? 그래서 다시 no_grad
        with torch.no_grad():
            # 평가에서는 제대로 소프트맥스 확률 필요없고
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        # 맞은 거 갯수, 총 갯수 누적해서
        correct += torch.sum(compare)
        total_examples += len(compare)

    # 나눠주면 정확도...
    return (correct / total_examples).item()


print(compute_accuracy(model, train_loader))

""" A.8 모델 저장했다 불러오기 """

# 저장은 그냥 간단하게 state_dict에 경로만 주면 끝이고..덮어쓴다...
torch.save(model.state_dict(), "data/model.pth")
# 새 모델은 당연히 기존 모델과 구조가 같아야 하고...
new_model = NeuralNetwork(2, 2)
# 불러올 때는 load_state_dict...
new_model.load_state_dict(torch.load("data/model.pth", weights_only=True))
