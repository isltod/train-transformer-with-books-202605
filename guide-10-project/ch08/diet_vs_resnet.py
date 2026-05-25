import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 여기부터 저 아래까지는 7장 vit_eye_doc 파일과 같은 코드...
DATA_DIR = "../../data/dataset/"
CLASSES = ["normal", "cataract", "glaucoma", "retina_disease"]

data = []
for class_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        data.append([img_path, class_idx])

df = pd.DataFrame(data, columns=["image_path", "label"])
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


train_dataset = CataractDataset(
    train_df["image_path"].values, train_df["label"].values, transform=train_transforms
)
test_dataset = CataractDataset(
    test_df["image_path"].values, test_df["label"].values, transform=test_transforms
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

data_iter = iter(train_loader)
images, labels = next(data_iter)


# 역정규화(unnormalize)를 수행해서 텐서 이미지를 넘파이 어레이로 변환
def imshow(img_tensor):
    img = img_tensor.numpy()
    # 트랜스포즈(transpose)
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


# # 이미지와 레이블 디스플레이
# fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
#
# for idx, (image, label) in enumerate(zip(images, labels)):
#     axes[idx].imshow(imshow(image))
#     axes[idx].set_title(f"Label: {label.item()}")
#     axes[idx].axis("off")
#
# plt.show()


def train(model, device, train_loader, optimizer, criterion, epoch, accelerator):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")


from sklearn.metrics import confusion_matrix, recall_score, precision_score


def test(model, device, test_loader, criterion, accelerator):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            output_cpu = output.to("cpu")
            target_cpu = target.to("cpu")
            pred = output_cpu.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_cpu.view_as(pred)).sum().item()

            all_preds.extend(pred.flatten().tolist())
            all_targets.extend(target.flatten().tolist())

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 혼동 행렬(confusion matrix), 리콜(sensitivity) 및 specificity 계산
    cm = confusion_matrix(all_targets, all_preds)
    sensitivity = recall_score(all_targets, all_preds, average=None)
    specificity = (cm.sum(axis=0) - cm.diagonal()) / cm.sum(axis=0)

    for i, (sens, spec) in enumerate(zip(sensitivity, specificity)):
        print(f"Class {i}: Sensitivity (Recall): {sens:.4f}, Specificity: {spec:.4f}")


# 여기까지 7장 vit_eye_doc과 같은 코드...
import timm

# 요기만 다른거 같다...모델이 vit가 deit로 바뀐것만...
model = timm.create_model("deit_base_patch16_224", pretrained=True)

# 클래스 수와 매칭하기 위해 최종 선형 층의 출력 사이즈 조정
num_classes = 4
model.head = torch.nn.Linear(model.head.in_features, num_classes)

from accelerate import Accelerator
from torch.optim import Adam

accelerator = Accelerator()
device = accelerator.device
learning_rate = 1e-4

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
model, optimizer, criterion = accelerator.prepare(model, optimizer, criterion)

# # 해보니 나는 정확도 75%...
# num_epochs = 10
# for epoch in range(1, num_epochs + 1):
#     train(model, device, train_loader, optimizer, criterion, epoch, accelerator)
#     test(model, device, test_loader, criterion, accelerator)

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ResNet-50 모델 불러오기
resnet_model = timm.create_model("resnet50", pretrained=True)

# 클래스 수와 매칭하기 위해 ResNet-50 최종 선형 층의 출력 사이즈 조정
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)

from accelerate import Accelerator
from torch.optim import Adam

accelerator = Accelerator()
device = accelerator.device
learning_rate = 1e-4

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
resnet_model, optimizer, criterion = accelerator.prepare(
    resnet_model, optimizer, criterion
)
# 이건 해보니 정확도가 12%밖에 안나온다...
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(resnet_model, device, train_loader, optimizer, criterion, epoch, accelerator)
    test(resnet_model, device, test_loader, criterion, accelerator)
