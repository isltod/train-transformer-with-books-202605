from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("klue", "ynat")
pprint(dataset)

raw_train_dataset = dataset["train"]
pprint(raw_train_dataset[0])

ratio = 80
dataset = load_dataset(
    path="csv",
    data_files="data/sample.csv",
    # 문자열로 슬라이싱한다...
    split={
        "train": f"train[:{ratio}%]",
        "test": f"train[{ratio}%:]",
    },
)
print(dataset)
