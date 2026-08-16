import tiktoken
import torch
from torch.utils.data import Dataset
from get_data import format_input


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        # 각 데이터 항목들에 대해서...
        for entry in data:
            # 지시, 입력, 출력 형식 맞추고 합쳐서...
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # 토큰 ID 목록으로 인코딩...
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# 일단 이건 입력만 처리하는 함수라서 정답지도 같이 처리하는 함수가 필요하다고...
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    # 배치에서 가장 긴 시퀀스 찾기...토큰을 하나 더 추가하는 이유는 정답지 쉬프트 처리와 관련있다..
    batch_max_length = max(len(item) + 1 for item in batch)

    # 입력 패딩 및 준비
    inputs_lst = []

    for item in batch:
        # 이것도 item을 밑에서 다시 사용하지 않는거 같은데 왜 굳이 copy해서 사용하지?
        new_item = item.copy()
        # <|endoftext|> 토큰 1개를 더 추가하고...
        new_item += [pad_token_id]
        # 정작 batch_max_length까지 시퀀스 패딩은 여기서 한다...
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 입력은 padded[:-1]를 통해 뭔 그 패딩을 제거한다...
        # 나중에 정답지는 한 칸 오른쪽으로 쉬프트 된 문장이 필요하므로 사용한다...
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # 입력 리스트를 텐서로 변환하고 타깃 장치로 전송
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


# 정답지는 따로 주는게 아니라 입력에서 오른쪽 한 칸 쉬프트...
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    # 배치에서 가장 긴 시퀀스 찾기 - 정답지 쉬프트 때문에 하나 더 길게...
    batch_max_length = max(len(item) + 1 for item in batch)

    # 입력 및 타깃 준비
    inputs_lst, targets_lst = [], []

    for item in batch:
        # 이것도 밑에서 사용하는 것도 아닌데 왜 copy하는지 모르겠다..
        new_item = item.copy()
        # <|endoftext|> 토큰 추가
        new_item += [pad_token_id]
        # 시퀀스를 max_length까지 패딩
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 입력을 위해 마지막 토큰 자르기
        targets = torch.tensor(padded[1:])  # 타깃을 위해 오른쪽으로 +1 이동
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 입력 리스트를 텐서로 변환하고 타깃 장치로 전송
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    # 배치에서 가장 긴 시퀀스 찾기 - 정답지 쉬프트 때문에 하나 더 길게...
    batch_max_length = max(len(item) + 1 for item in batch)

    # 입력과 타깃 패딩 및 준비
    inputs_lst, targets_lst = [], []

    for item in batch:
        # 이것도 밑에서 사용하는 것도 아닌데 왜 copy하는지 모르겠다..
        new_item = item.copy()
        # <|endoftext|> 토큰 추가
        new_item += [pad_token_id]
        # 시퀀스를 max_length까지 패딩
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 입력을 위해 마지막 토큰 자르기
        targets = torch.tensor(padded[1:])  # 목표를 위해 오른쪽으로 +1 이동

        # 새로 추가: 목표에서 첫 번째 패딩 토큰을 제외한 모든 토큰을 ignore_index로 바꾸기
        # 일단 패딩 ID인 부분을 True = 1로 하고 거기 인덱스를 찾아서...
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # 각 줄에서 패딩이 2개 이상 들어있으면...
        if indices.numel() > 1:
            # 각 줄에서 첫 번째 <|endoftext|> 만 빼고 나머지 인덱스 자리는 다 -100
            # 첫 번째는 문장 마침으로 학습해야 해서 남긴다고...
            targets[indices[1:]] = ignore_index

        # 새로 추가: 최대 시퀀스 길이로 자르기 (선택 사항)
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 입력 및 타깃 리스트를 텐서로 변환하고 타깃 장치로 전송
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    # custom_collate_draft_1 테스트
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    print(custom_collate_draft_1(batch))

    # custom_collate_draft_2 테스트
    inputs, targets = custom_collate_draft_2(tokenizer)
    print(inputs)
    print(targets)

    # 최종 custom_collate_fn 테스트
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    # 그럼 왜 ignore_index에 -100을 쓰나?
    # 1. 우선 예로 기본 형의 크로스 엔트로피는 1.1269
    logits_1 = torch.tensor(
        [[-1.0, 1.0], [-0.5, 1.5]]  # 첫 번째 훈련 샘플  # 두 번째 훈련 샘플
    )
    targets_1 = torch.tensor([0, 1])
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)
    # 2. 여기에 뭐든 데이터를 하나 더 추가하면 값이 바뀌는데...
    logits_2 = torch.tensor(
        [[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]]  # 새로운 세 번째 훈련 샘플
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)
    # 3. 추가한 데이터의 정답을 -100으로 하면 세 번째를 무시하므로 값이 같아진다...
    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)
    # 이유는 파이토치의 크로스 엔트로피 함수가 cross_entropy(..., ignore_index=-100)라서...
