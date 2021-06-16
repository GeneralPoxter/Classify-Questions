"""
Developer: Damian Rene
Comments: Based on code by Atith Gandhi
"""

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
)
from nlp import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

dirname = os.path.dirname(__file__)

model = BertForSequenceClassification.from_pretrained(
    os.path.join(dirname, "../models/bert-base-cased")
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        padding="max_length",
        return_attention_mask=True,
    )


# Setup testing dataset
test_dataset = load_dataset(
    "json",
    data_files={"test": os.path.join(dirname, "../data/qanta_test.json")},
    field="questions",
)["test"]
test_dataset = test_dataset.map(
    lambda example: {"label": [0 if example["difficulty"] == "hs" else 1]}
)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Test model
def compute_metrics(pred):
    labels = pred.label_ids
    # print(labels)
    preds = pred.predictions.argmax(-1)
    # print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model, compute_metrics=compute_metrics, eval_dataset=test_dataset
)

print(trainer.evaluate())
