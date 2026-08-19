import os
import sys
import json

# GTX 960(2GB/Maxwell)으로 인한 메모리 액세스 예외 및 드라이버 꼬임 방지를 위해
# 고성능 GPU인 RTX 3090(Device 0)만 활성화
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import time
import numpy as np
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

# 이전 장 유틸리티 함수 import
from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


# ==========================================
# 1. 데이터 다운로드 & 전처리
# ==========================================
def download_and_load_file(file_path):
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def split_data(data):
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    return train_data, val_data, test_data


# ==========================================
# 2. Dataset & Collate Function
# ==========================================
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# ==========================================
# 3. 모델 초기화
# ==========================================
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # PyTorch와의 DLL/VRAM 충돌 방지를 위한 TensorFlow 지연 로딩
    # 이게 gpt_download.py 글로벌 import 부분에 있는 것이 문제의 원인 후보라고...
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def init_pretrained_model(choose_model):
    cfg = BASE_CONFIG.copy()
    cfg.update(model_configs[choose_model])

    model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
    models_dir = "gpt2"
    model_dir = os.path.join(
        "e:\\Devs\\train-transformer-with-books-202605\\data\\", models_dir, model_size
    )

    # gpt_download의 download_and_load_gpt2 로직 사용
    from gpt_download import download_file

    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    import tensorflow as tf

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(
        open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8")
    )
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    return model, cfg


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            eos_id=50256,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


# ==========================================
# 5. 훈련 루프
# ==========================================
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    device = accelerator.device
    print(f"사용 장치: {device}")
    if device.type == "cuda":
        print("현재 GPU 번호:", torch.cuda.current_device())
        print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            accelerator.backward(loss)
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"에포크 {epoch+1} (Step {global_step:06d}): "
                    f"훈련 손실 {train_loss:.3f}, 검증 손실 {val_loss:.3f}"
                )

        print(f"\n--- 에포크 {epoch+1} 완료 샘플 생성 ---")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print("-" * 50)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()


# ==========================================
# 6. 메인 실행 루프
# ==========================================
def main():
    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer = tiktoken.get_encoding("gpt2")

    # 1) 데이터 로드 및 분할
    file_path = "instruction-data.json"
    data = download_and_load_file(file_path)
    train_data, val_data, test_data = split_data(data)

    print(
        f"훈련 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개, 테스트 데이터: {len(test_data)}개"
    )

    # 2) 데이터로더 구성 (custom_collate_fn 적용)
    customized_collate_fn = partial(
        custom_collate_fn, device=accelerator.device, allowed_max_length=1024
    )

    batch_size = 8
    num_workers = 0

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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

    # 3) 사전 학습된 모델 로드
    CHOOSE_MODEL = "gpt2-medium (355M)"
    print(f"선택한 모델: {CHOOSE_MODEL} 로드 중...")
    model, cfg = init_pretrained_model(CHOOSE_MODEL)

    torch.manual_seed(123)

    # 4) 옵티마이저 및 Accelerator prepare
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 훈련 전 손실 확인
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, accelerator.device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, accelerator.device, num_batches=5
        )

    print(f"훈련 전 손실 -> 훈련: {train_loss:.4f}, 검증: {val_loss:.4f}")

    # 5) 훈련 실행
    num_epochs = 2
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"\n훈련 소요 시간: {execution_time_minutes:.2f}분")

    # 6) 시각화
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 7) 샘플 응답 생성 및 테스트 세트 평가
    torch.manual_seed(123)
    model.eval()

    print("\n--- 샘플 응답 평가 (3개) ---")
    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(accelerator.device),
            max_new_tokens=256,
            context_size=cfg["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        print(f"\n[입력]:\n{input_text}")
        print(f"\n[정답 응답]:\n>> {entry['output']}")
        print(f"\n[모델 응답]:\n>> {response_text.strip()}")
        print("-" * 50)

    # 8) 전체 테스트 데이터 세트 응답 생성 및 json 저장
    print("\n전체 테스트 데이터 세트 생성 중...")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(accelerator.device),
            max_new_tokens=256,
            context_size=cfg["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )
        test_data[i]["model_response"] = response_text

    save_json_path = "instruction-data-with-response.json"
    with open(save_json_path, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)

    print(f"테스트 데이터 생성 결과가 {save_json_path}에 저장되었습니다.")

    # 9) 파인튜닝된 모델 체크포인트 저장
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    # Accelerator 모델 원본 구하기
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), file_name)
    print(f"모델 가중치가 {file_name}에 성공적으로 저장되었습니다.")


if __name__ == "__main__":
    main()
