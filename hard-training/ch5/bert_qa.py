from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 같은 토크나이저
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# ForQuestionAnswering 모델
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
print(model)

# 기계 독해란다...그래서 mrc?
from datasets import load_dataset

dataset = load_dataset("klue", "mrc")
sample = dataset["train"][0]

# 예문, 문제, 정답...수능이네...
print(f"내용 : {sample['context'][:50]}")
print(f"질문 : {sample['question']}")
print(f"답변 : {sample['answers']}")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
batch = data_collator([tokenized_dataset["train"][i] for i in range(10)])
print(batch)

import torch

with torch.no_grad():
    outputs = model(**batch)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = batch["input_ids"][0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
