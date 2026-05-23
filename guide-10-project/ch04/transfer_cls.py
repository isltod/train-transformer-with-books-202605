import pandas as pd
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# from transformers import AdamW  # 원서에서 사용된 코드이나 최근 이 기능이 제거되어 아래 코드로 대체함
from torch.optim import AdamW
import sys

sys.path.append("../../")
from wolf import get_my_gpu_device, Timer

device = get_my_gpu_device(0)

# 데이터는 캐글의 가짜 진짜 뉴스 데이터...
real = pd.read_csv("../../data/True.csv")
fake = pd.read_csv("../../data/Fake.csv")
real = real.drop(["title", "subject", "date"], axis=1)
real["label"] = 1.0
fake = fake.drop(["title", "subject", "date"], axis=1)
fake["label"] = 0.0
dataframe = pd.concat([real, fake], axis=0, ignore_index=True)
df = dataframe.sample(frac=0.1).reset_index(drop=True)
print(df.head(20))
print(len(df[df["label"] == 1.0]))
print(len(df[df["label"] == 0.0]))

# 데이터 준비는 토크나이저...모델 전용 토크나이저를 사용하라고...
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 학습/검증 데이터 분리
texts, labels = tuple(df["text"].tolist()), tuple(df["label"].tolist())
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2
)


# texts, lables 받아서 input_ids, attention_mask, labels_out 반환
def tokenize_and_encode(texts, labels):
    input_ids, attention_masks, labels_out = [], [], []
    for text, label in zip(texts, labels):
        # 여기서 전역 변수로 토크나이저를 사용하니 위에서 선언해야 된다...
        encoded = tokenizer(text, max_length=512, padding="max_length", truncation=True)
        # 트랜스포머 토크나이저를 쓰면 어텐션 마스크를 자동으로 만드네...
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels_out.append(label)
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_masks),
        torch.tensor(labels_out),
    )


# 토큰화
train_input_ids, train_attention_masks, train_labels = tokenize_and_encode(
    train_texts, train_labels
)
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
    val_texts, val_labels
)
print(
    "train_input_ids ",
    train_input_ids[0].shape,
    train_input_ids[0],
    "\n" "train_attention_masks ",
    train_attention_masks[0].shape,
    train_attention_masks[0],
    "\n" "train_labels",
    train_labels[0],
)


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels, num_classes=2):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.num_classes = num_classes
        # 그냥 labels도 있고 one_hot_labels도 있고...
        self.one_hot_labels = self.one_hot_encode(labels, num_classes)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.one_hot_labels[idx],
        }

    # 클래스 내에서 부르려면 정적 메서드여야 하나?
    @staticmethod
    def one_hot_encode(targets, num_classes):
        targets = targets.long()
        one_hot_targets = torch.zeros(targets.size(0), num_classes)
        # 1번 차원 - 클래스 수 방향으로,  targets의 숫자를 인덱스로 봐서 1.0을 넣는다...
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot_targets


train_dataset = TextClassificationDataset(
    train_input_ids, train_attention_masks, train_labels
)
val_dataset = TextClassificationDataset(val_input_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=4)
print(len(train_dataset))
print(len(val_dataset))
item = next(iter(train_dataloader))
item_ids, item_mask, item_labels = (
    item["input_ids"],
    item["attention_mask"],
    item["labels"],
)
print(
    "item_ids, ",
    item_ids.shape,
    "\n",
    "item_mask, ",
    item_mask.shape,
    "\n",
    "item_labels, ",
    item_labels.shape,
    "\n",
)


# 모델을 학습시키는데...시간을 좀 비교해보자...
def run_step(model, batch, accelerator, optimizer, lr_scheduler):
    # 훈련에서는 가속기가 알아서 데이터를 gpu로 보낸다네...
    outputs = model(**batch)
    loss = outputs.loss
    # 역전파 시동을 loss가 아니라 가속기에서 하네...
    accelerator.backward(loss)
    optimizer.step()
    # 알아서가 아니라 이렇게 직접 학습률을 감소시켜야 하는 모양...
    lr_scheduler.step()
    optimizer.zero_grad()


def train(grad_accum, train_dataloader, eval_dataloader):
    # 모델 및 옵티마이저 준비
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 가속기라고 해야하나? 아무튼 가속기를 통과시킨다...
    if grad_accum:
        accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")
    else:
        accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    # model.gradient_checkpointing_enable()

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    # 학습률을 조정해서 훈련이 수렴되도록 하는 객체쯤 되는 모양...
    lr_scheduler = get_scheduler(
        # 0으로 선형 감소
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # tqdm을 이렇게도 쓸 수 있구나..
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        # 데이터 종류별로 안받고 그냥 이렇게 통으로 받아서 **하는 수도 있구나...
        for batch in train_dataloader:
            if grad_accum:
                # 가속기에 gradient_accumulation_steps 사용하면 이걸로 감싸야 한다고...
                # 근데 안 감싸도 돌아가기는 하던데...
                with accelerator.accumulate(model):
                    run_step(model, batch, accelerator, optimizer, lr_scheduler)
            else:
                run_step(model, batch, accelerator, optimizer, lr_scheduler)
                # 이건 전체 중 1번 했다는 건가?
            progress_bar.update(1)

        # 평가
        model.eval()
        preds = []
        out_label_ids = []
        epochs = 1
        epoch = 1

        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits

            preds.extend(torch.argmax(logits.detach().cpu(), dim=1).numpy())
            out_label_ids.extend(
                torch.argmax(inputs["labels"].detach().cpu(), dim=1).numpy()
            )
        accuracy = accuracy_score(out_label_ids, preds)
        f1 = f1_score(out_label_ids, preds, average="weighted")
        recall = recall_score(out_label_ids, preds, average="weighted")
        precision = precision_score(out_label_ids, preds, average="weighted")

        # 일단 평가 데이터에 대해서는 엄청 점수가 좋다...
        print(f"Epoch {epoch + 1}/{num_epochs} Evaluation Results:")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        return model


# gradient_accumulation_steps를 사용하는게 훨씬 빠르다...
with Timer():
    model = train(True, train_dataloader, eval_dataloader)

# 여기도 추론에서는 토크나이저를 새로 만드는데...그것도 Auto가 아니라 Bert로 만드네?
# 실제 일할 때 상황을 보여주려는 건가?
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def inference(text, model, label, device=device):
    # 토크나이저 불러오기 및 입력 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 입력 텐서를 특정 디바이스로 전송
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델을 eval 모드로 설정 후 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # predicted label 인덱스 추출 - 이건 cpu로 보내야...
    pred_label_idx = torch.argmax(logits.detach().cpu(), dim=1).item()

    print(f"Predicted label index: {pred_label_idx}, actual label {label}")
    return pred_label_idx


# 책은 다 맞췄다고 하는데, 나는 틀리는데? 진짜 기사도 가짜라는데?
text = """
WASHINGTON (ABC) A confirmed tornado was located near Bridgeville in Sussex County, Delaware, shortly after 6 p.m. ET Saturday, moving east at 50 mph, according to the National Weather Service. Downed trees and wires were reported in the area.
"""
inference(text, model, 1.0)
text = "this is definately junk text I am typing"
inference(text, model, 0.0)
