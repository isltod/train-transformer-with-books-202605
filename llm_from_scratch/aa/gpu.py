import torch

print(torch.__version__)
print(torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())

# 계산은 기본적으로 CPU
tensor_1 = torch.tensor([1.0, 2.0, 3.0])
tensor_2 = torch.tensor([4.0, 5.0, 6.0])
print(tensor_1 + tensor_2)

# to를 이용해서 GPU로 옮기면 GPU에서 연산
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
print(tensor_1 + tensor_2)
# 연산 전에 잠깐 멈춤 발생하고 결과도 cuda로 나온다...
# 당연히 연산에 필요한 모든 데이터는 같은 장치에 있어야 한다..

# GPU 연산으로 훈련하기 - 다 똑같고 세 군데만 바뀐다
# 데이터가 있고,
X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test = torch.tensor([0, 1])

from torch.utils.data import Dataset


# 데이터셋으로 받고
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

from torch.utils.data import DataLoader

torch.manual_seed(123)
# 그걸 다시 데이터로더로 감싸고...
# 주의! num_workers를 1 이상으로 주면 하위 프로세스가 활성화되서 __main__ 블럭으로 감싸지 않으면 예외 발생
train_loader = DataLoader(
    dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)


# 모델 네트워크까지 기본 준비는 다 같고
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

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


import torch.nn.functional as F

# 1. 장치를 불러온다
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(num_inputs=2, num_outputs=2)
# 2. 모델을 GPU로 보내고
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        # 데이터를 장치로 보낸다 - 그럼 알아서 GPU 연산으로...
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)  # Loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

    model.eval()
    # Optional model evaluation


def compute_accuracy(model, dataloader, device):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        # 모델을 평가할 때로 데이터는 GPU로 보낸다...
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


print(compute_accuracy(model, train_loader, device=device))
