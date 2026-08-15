import json
import os
import requests


def download_and_load_file(file_path):
    # 이 URL은 어차피 고정인데 입력으로 사용하기 싫어서 함수 내 상수로 고정...
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    # 파일이 없는 경우만 새로 받는다...
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# 알파카 프롬프트 스타일 만들기
def format_input(entry):
    # 지시문 포멧 만들고
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    # 입력 문 형식 만들고
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    # 합쳐서 반환...
    return instruction_text + input_text


# 데이터 나누기
def split_data(data):
    train_portion = int(len(data) * 0.85)  # 훈련을 위한 85%
    test_portion = int(len(data) * 0.1)  # 테스트를 위한 10%
    val_portion = len(data) - train_portion - test_portion  # 나머지 5%는 검증용

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    print("훈련 세트 길이:", len(train_data))
    print("검증 세트 길이:", len(val_data))
    print("테스트 세트 길이:", len(test_data))

    return train_data, val_data, test_data


if __name__ == "__main__":
    file_path = "instruction-data.json"

    data = download_and_load_file(file_path)
    print("샘플 개수:", len(data))
    print("샘플 예시:\n", data[50])
    print("다른 샘플:\n", data[999])

    # 입력 필드가 있는 경우 출력
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)
    # 입력 필드가 없는 경우...
    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)

    split_data(data)
