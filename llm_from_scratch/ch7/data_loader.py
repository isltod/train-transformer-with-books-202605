import tiktoken
import torch
from functools import partial
from custom_collate import custom_collate_fn, InstructionDataset
from torch.utils.data import DataLoader
from get_data import download_and_load_file, split_data


def get_loader(device):
    # 앞에서 정의했던 custom_collate_fn에 장치와 최대 길이를 고정시키는 함수...필요한가?
    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    # 앞에서 받아서 만들었던 지시 데이터셋...download 부분은 있으면 넘어가기는 하는데...
    file_path = "instruction-data.json"
    data = download_and_load_file(file_path)
    train_data, val_data, test_data = split_data(data)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    # 데이터 로드 만들기 - 토크나이저 -> 데이터셋 -> 데이터로더
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # 이전과 다르게 여기에 지시 fine tuning에 맞는 함수를 넣어준다...
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, val_data


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # 파이토치 2.9 이상에서는 mps 결과가 안정적입니다.
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("장치:", device)
    train_loader, _, _, _ = get_loader(device)

    print("훈련 데이터 로더:")
    for inputs, targets in train_loader:
        # 배치는 8, 그 뒤에 문장 길이는 다 다르다...
        print(inputs.shape, targets.shape)
    # 입력은 마지막에 50256 <|endoftext|>로 끝나야 하고, 정답지는 남는 부분이 -100으로...
    print(inputs[0])
    print(targets[0])
