from transformers import AutoTokenizer, AutoModelForMultipleChoice

model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)
print(model)

from datasets import load_dataset

# dataset = load_dataset("HAERAE-HUB/csatqa", "full")
# dataset = load_dataset("EleutherAI/CSAT-QA", "full")
dataset = load_dataset(
    "EleutherAI/CSAT-QA", "GR"
)  # Choose from either WR, GR, LI, RCH, RCS, RCSS,


print(dataset["test"][0])

ending_names = ["option#1", "option#2", "option#3", "option#4", "option#5"]


def preprocess_function(examples):
    first_sentences = [[context] * 5 for context in examples["context"]]
    question_headers = examples["question"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names]
        for i, header in enumerate(question_headers)
    ]
    # 토큰화를 위해 1차원으로 평활화
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # None 데이터 처리
    first_sentences = [i if i else "" for i in first_sentences]
    second_sentences = [i if i else "" for i in second_sentences]

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)

    # 토큰화 후 다시 2차원으로 재배열
    result = {
        k: [v[i : i + 5] for i in range(0, len(v), 5)]
        for k, v in tokenized_examples.items()
    }
    result["labels"] = [
        i - 1 for i in examples["gold"]
    ]  # 원활한 collator 사용을 위한 변수명 이동, 레이블 0번부터 시작하게 변경

    return result


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["test"].column_names
)

from dataclasses import dataclass
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
batch = collator([tokenized_dataset["test"][i] for i in range(5)])

with torch.no_grad():
    logits = model(**batch).logits
print(logits)
