# 오토클래스
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# 비슷한 일을 할 때, 세부적인 변형은 알아서 찾아주는 AutoClass라고...
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
    as_tuple=True
)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print("모델이 내놓은 답은", tokenizer.decode(predicted_token_id))
