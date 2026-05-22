from transformers import BertTokenizer

# 일단 이렇게 간단하게 from_pretrained로 불러온다...
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.tokenize("The tokenizers"))
