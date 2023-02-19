import shutil

import fire
import joblib
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "prajjwal1/bert-mini"


class TextClassifierDataset(Dataset):
    def __init__(self, data, l_enc, tokenizer):
        self.data = data
        self.l_enc = l_enc
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        title = self.data.loc[idx, "title"]
        abstract = self.data.loc[idx, "abstract"]
        tasks = self.data.loc[idx, "tasks"]
        tasks_v = self.l_enc.transform([tasks])[0, :]

        item = self.tokenizer(
            f"{title} {abstract}", padding="max_length", truncation=True, max_length=512
        )

        item = {key: torch.tensor(val) for key, val in item.items()}

        item["labels"] = torch.from_numpy(tasks_v).float()

        return item


def train_model(train_path, val_path, mlb_path, model_path):
    train = pd.read_json(train_path)
    val = pd.read_json(val_path)

    mlb = joblib.load(mlb_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    train_dataset = TextClassifierDataset(data=train, l_enc=mlb, tokenizer=tokenizer)
    val_dataset = TextClassifierDataset(data=val, l_enc=mlb, tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        problem_type="multi_label_classification",
        num_labels=len(mlb.classes_),
        label2id={x: i for i, x in enumerate(mlb.classes_)},
        id2label={i: x for i, x in enumerate(mlb.classes_)},
    )

    training_arguments = TrainingArguments(
        output_dir="/tmp/",
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=30,
        load_best_model_at_end=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    shutil.move(trainer.state.best_model_checkpoint, model_path)

    return trainer.state.best_model_checkpoint


if __name__ == "__main__":
    fire.Fire(train_model)
