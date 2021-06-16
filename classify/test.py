"""
Developer: Damian Rene
Comments: Based on code by Atith Gandhi
"""

from transformers import Trainer
from nlp import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

DIR = os.path.dirname(__file__)
TEST_PATH = os.path.join(DIR, "../data/qanta_test.json")


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


def test(model, model_name, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
        )

    test_dataset = load_dataset(
        "json", data_files={"test": TEST_PATH}, field="questions"
    )["test"]
    test_dataset = test_dataset.map(
        lambda example: {"label": [0 if example["difficulty"] == "hs" else 1]}
    )
    test_dataset = test_dataset.map(
        tokenize, batched=True, batch_size=len(test_dataset)
    )
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(
        model=model, compute_metrics=compute_metrics, eval_dataset=test_dataset
    )

    print(f"{model_name}: {trainer.evaluate()}")
