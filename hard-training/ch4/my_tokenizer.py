from pprint import pprint
from datasets import load_dataset
from transformers import BertTokenizerFast

path = "e:/Devs/train-transformer-with-books-202605/"

# 1. 데이터 재료 로드하고, title만 추리고, 특수 토큰 만들고...
dataset = load_dataset("klue", "ynat")
pprint(dataset["train"][0])
pprint(dataset.column_names)

# title만 한글 문장이니까 그걸 토큰화 연습...메모릴 아낀다고 타이틀을 파일로 저장한다...
target_key = "title"
for key in dataset.column_names.keys():
    with open(f"{path}data/tokenizer_data_{key}.txt", "w", encoding="utf-8") as f:
        # key는 train과 validation, dataset[key][target_key]까지 가면 Column 개체?로 리스트? 배열? 뭐 그런 식...
        f.write("\n".join(dataset[key][target_key]))

user_defined_symbols = [
    "[PAD]",  # 문장의 길이를 맞추기 위해 사용되는 토큰
    "[UNK]",  # 토크나이저가 인식할 수 없는 토큰
    "[CLS]",  # bert 계열 모델에서 문장 전체의 정보를 저장하는 토큰
    "[SEP]",  # bert 계열 모델에서 문장 구분을 위해 사용하는 토큰
    "[MASK]",  # MLM 모델에서 토큰 마스킹을 위해 사용하는 토큰
]

unused_token_num = 100
unused_list = [
    f"[UNUSED{i}]" for i in range(unused_token_num)
]  # 사전학습 시, 어휘에 없는 토큰을 추가하기 위한 빈 공간

whole_user_defined_symbols = user_defined_symbols + unused_list
pprint(whole_user_defined_symbols[:10])

# 2. 빈 토크나이저 만들고
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

# bert는 WordPiece를 기반으로 하는 모델이다...
bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))


# 3. 정규화를 위해 normalizer를 만들어 끼워넣고...
from tokenizers import normalizers

# clean_text, 중국어 처리, 알파벳 포함 심볼 제거, 소문자로 변경...이런게 정규화...
normalizer = normalizers.BertNormalizer()
bert_tokenizer.normalizer = normalizer
pprint(normalizer.normalize_str("Héllò hôw\nare ü? "))

# 4. 사전 토크나이저를 끼워넣고...
# 워드피스 이전에 먼저 적용되는 간단한 토크나이저? 왜 두 개나 써야 하지?
from tokenizers.pre_tokenizers import Whitespace

# 공백, 줄바꿈으로 자르는 토크나이저...
pre_tokenizer = Whitespace()
bert_tokenizer.pre_tokenizer = pre_tokenizer
pprint(
    pre_tokenizer.pre_tokenize_str("안녕하세요. 제대로 인코딩이 되는지 확인 중입니다.")
)

# 5. 사후 처리기 끼워넣고...
from tokenizers.processors import TemplateProcessing

# 문장이 인코딩되면 어떤 형태가 되야 하는지 양식을 지정한다?
post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[(t, i) for i, t in enumerate(user_defined_symbols)],
)
bert_tokenizer.post_processor = post_processor

# 6. 학습기 만들고...
from tokenizers.trainers import WordPieceTrainer

vocab_size = 24000
trainer = WordPieceTrainer(
    vocab_size=vocab_size,
    special_tokens=whole_user_defined_symbols,
)

# 7. 이제 진짜 학습
from glob import glob

# 막상 학습은 자체 train 메서드로 한다..
bert_tokenizer.train(glob(f"{path}data/tokenizer_data_*.txt"), trainer)

# 8. 결과 확인
output = bert_tokenizer.encode("인코딩 및 디코딩이 제대로 이루어지는지 확인 중입니다.")
# 인코딩은 잘 된다...
print(output.ids)
# 디코딩은 이상하게 보이는데...
print(bert_tokenizer.decode(output.ids))

# 토크나이저에 원래 기반인 워드피스 디코더를 끼워넣고 해보면 잘 된다고...
from tokenizers import decoders

bert_tokenizer.decoder = decoders.WordPiece()
print(bert_tokenizer.decode(output.ids))

# 9. 학습된 토크나이저를 transformers의 토크나이저로 만들고
# from transformers import BertTokenizerFast

# 적당한 토크나이저 클래스를 선택하고, 생성자에서 tokenizer_object를 지정하면 되는 모양...
fast_tokenizer = BertTokenizerFast(tokenizer_object=bert_tokenizer)
encoded = fast_tokenizer.encode("인코딩 및 디코딩이 제대로 이루어지는지 확인 중입니다.")
decoded = fast_tokenizer.decode(encoded)
print(encoded)
print(decoded)

# 10. 다음에 사용하려면 저장
output_dir = path + "data/MyTokenizer"
fast_tokenizer.save_pretrained(output_dir)

# 11. 다음에, 다른 곳에서 사용한다면 불러와서 사용...
new_tokenizer = BertTokenizerFast.from_pretrained(output_dir)

encoded = new_tokenizer(["인코딩 잘 되는지 확인", "안되면 다시 학습하자"])

for k, v in encoded.items():
    print(k, v)

print(new_tokenizer.decode(encoded["input_ids"][0]))
print(new_tokenizer.decode(encoded["input_ids"][1]))
