from datasets import load_dataset

# 1. 아예 파이토치 텐서로 데이터를 불러온다...
imdb_dataset_torch = load_dataset("imdb").with_format("torch")
print(type(imdb_dataset_torch["train"]))

from datasets import DatasetInfo
from torch.utils.data import Dataset

print(DatasetInfo(imdb_dataset_torch))

imdb_train = imdb_dataset_torch["train"]
# 딕셔너리지만 번호로 인덱싱...
print(imdb_train[0])

# 데이터를 불러와서 torch.utils.data.Dataset으로 만든다...
imdb_dataset = load_dataset("imdb")


# 데이터셋을 전처리한 후 딕셔너리 리스트 반환 - text와 label을 포함
def preprocess(data):
    dataset = []
    for example in data:
        text = example["text"].lower()
        label = example["label"]
        dataset.append({"text": text, "label": label})
    return dataset


# 학습 데이터 생성
train_data = preprocess(imdb_dataset["train"])


# torch.utils.data.Dataset을 상속하는 CustomDataset 클래스 생성
class CustomDataset(Dataset):
    # 생성자에서 data에 넣어준다...
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = CustomDataset(train_data)
print(train_dataset[0])
