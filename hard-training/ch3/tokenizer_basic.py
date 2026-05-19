from transformers import BertTokenizer

# 일단 책에는 없고 소스에만 있는 테스트 코드...
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="data")

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
print("디코딩 결과:")
decoded_string1 = tokenizer.decode(ids1)
print(decoded_string1)

decoded_string2 = tokenizer.decode(ids2["input_ids"])
print(decoded_string2)

decoded_string3 = tokenizer.decode(ids2["input_ids"], skip_special_tokens=True)
print(decoded_string3)
