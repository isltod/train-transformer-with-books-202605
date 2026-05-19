from transformers import BertTokenizer

# 1. 토크나이저를 다운받은 모델에서 생성하고
# 모델을 다운로드 한다는 거겠지?
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
# 리눅스 man 페이지처럼 끝없이 도움말이 나오네...
# print(help(tokenizer))

print(tokenizer.vocab_size)
# print(tokenizer.get_vocab())
print(tokenizer.special_tokens_map)

sentence = "안녕하세요. 이건 테스트입니다."

# 토큰화 작업
tokens1 = tokenizer.tokenize(sentence)
print("토큰화 작업 결과:")
print(tokens1)

# 토큰을 입력 식별자로 변환
ids1 = tokenizer.convert_tokens_to_ids(tokens1)
print(ids1)

ids2 = tokenizer(sentence)
print(ids2)

# 디코딩
decoded_string1 = tokenizer.decode(ids1)
print(decoded_string1)

decoded_string2 = tokenizer.decode(ids2["input_ids"])
print(decoded_string2)

decoded_string3 = tokenizer.decode(ids2["input_ids"], skip_special_tokens=True)
print(decoded_string3)

# 2. 데이터셋을 다운받고 토큰화...
from datasets import load_dataset

dataset = load_dataset("klue", "ynat")
raw_train_dataset = dataset["train"]

# 이 코드는 ValueError...
# tokenized_examples = tokenizer(
#     raw_train_dataset["title"],
#     padding="max_length",
#     truncation=True,
# )
# print(tokenized_examples)


def tokenize_function(sample):
    return tokenizer(sample["title"])


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["guid", "title", "url", "date"],
)
print(tokenized_datasets)

print(tokenized_datasets["train"][0]["input_ids"])
print(type(tokenized_datasets["train"][0]["input_ids"]))

# 3. 뭔진 모르겠는데 collator라는 걸 하고...
from pprint import pprint
from transformers import DataCollatorWithPadding

batch = [tokenized_datasets["train"][i] for i in range(8)]
print([len(sample["input_ids"]) for sample in batch])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(batch)
pprint({k: v.size() for k, v in batch.items()})

# 4. 모델을 다운받고
from transformers import BertTokenizer, BertModel

# 이건 맨 위에서 했던거와 같은데...모델만 다른데...아무튼 받아서 저장하는 실습인가?
model = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model)
model = BertModel.from_pretrained(model)

model_path = "data/MyBertModel/"
# 토크나이저와 모델을 각각 저장해야 하는 모양...
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

# 다시 생성하면 캐시에서 불러오겠지...
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 5. 추론 작업을 한다...
import torch
from transformers import BertTokenizer, BertForMaskedLM

model = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model)
model = BertForMaskedLM.from_pretrained(model)

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
    as_tuple=True
)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print("모델이 내놓은 답은", tokenizer.decode(predicted_token_id))


# 직접구현
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "google-bert/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.cuda().eval()

with torch.no_grad():
    output = model(
        **tokenizer(
            "유튜브 내달 2일까지 크리에이터 지원 공간 운영", return_tensors="pt"
        ).to(model.device)
    )
    result = torch.softmax(output.logits.cpu(), -1)

result = [
    {"label": f"LABEL_{l}", "score": result[i, l].item()}
    for i, l in enumerate(result.argmax(-1))
]
print(result)
