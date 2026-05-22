import os

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# from trl import KTOTrainer, KTOConfig
from trl.experimental.kto import KTOConfig, KTOTrainer

model_name = "facebook/opt-350m"
ref_model_name = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{message['role'] + ': ' + message['content'] + '\n\n'}}"
        "{% endfor %}{{ eos_token }}"
    )

dataset = load_dataset("trl-lib/kto-mix-14k", split="train")


def process(row):
    global tokenizer
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
    row["completion"] = tokenizer.apply_chat_template(row["completion"], tokenize=False)
    return row


if __name__ == "__main__":
    print(dataset)
    dataset = dataset.map(
        process,
        num_proc=4,
        load_from_cache_file=False,
    )
    print(dataset[0]["completion"])

    args = KTOConfig(
        logging_dir="../../data/logs",
        output_dir="../../data/ckpt",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        optim="adamw_torch",
        logging_steps=100,
        report_to="none",
        max_length=512,
        # 여기도 이건 없어졌다고...
        # max_prompt_length=512,
        remove_unused_columns=False,
        dataset_num_proc=2,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
    )

    trainer = KTOTrainer(
        model,
        ref_model,
        args=args,
        train_dataset=dataset,
        # 이것도 계속되는데...
        # tokenizer=tokenizer,
        processing_class=tokenizer,
    )

    trainer.train()
