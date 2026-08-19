import re
import torch
import time
from previous_chapters import calc_loss_loader, train_model_simple, plot_losses
from load_pretrained import device, init_pretrained_model
from data_loader import get_loader

if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-medium (355M)"
    model = init_pretrained_model(CHOOSE_MODEL)
    model.to(device)

    torch.manual_seed(123)

    train_loader, val_loader, test_loader, _ = get_loader(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("훈련 손실:", train_loss)
    print("검증 손실:", val_loss)

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"훈련 소요 시간: {execution_time_minutes:.2f}분")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        print(input_text)
        print(f"\n올바른 응답:\n>> {entry['output']}")
        print(f"\n모델 응답:\n>> {response_text.strip()}")
        print("-------------------------------------")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # 미려한 출력을 위해 "indent" 사용

    print(test_data[0])

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"모델이 {file_name}에 저장되었습니다.")

    # 모델 로드 방법:
    # model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
