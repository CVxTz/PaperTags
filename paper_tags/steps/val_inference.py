from pathlib import Path

import fire
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

MODEL = "prajjwal1/bert-mini"


def predict_val(val_path, model_path, output_path):
    val = pd.read_json(val_path)

    pipe = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=AutoTokenizer.from_pretrained(MODEL),
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    val["predictions"] = pipe(
        [f"{title} {abstract}" for title, abstract in zip(val.title, val.abstract)],
        top_k=10,
        batch_size=32,
        max_length=512,
        truncation=True,
    )

    val.to_json(
        Path(output_path) / "val.json", orient="records", indent=2, force_ascii=False
    )


if __name__ == "__main__":
    fire.Fire(predict_val)
