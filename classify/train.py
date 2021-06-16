"""
Developer: Damian Rene
Comments: Based on code by Atith Gandhi
"""

from transformers import Trainer, TrainingArguments
from nlp import load_dataset
import os

DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(DIR, "../data/qanta_train.json")

training_args = TrainingArguments(
    output_dir="",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # warmup_steps = 0,
    # weight_decay=1e-8,
    learning_rate=1e-5,
    # evaluate_during_training=True,
    save_steps=500,
    logging_steps=500,
    do_eval=True,
)


def train(model, model_name, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
        )

    training_args.output_dir = os.path.join(DIR, f"../results/{model_name}")
    training_args.logging_dir = os.path.join(DIR, f"../logs/{model_name}")

    train_dataset = load_dataset(
        "json", data_files={"train": TRAIN_PATH}, field="questions"
    )["train"]
    train_dataset = train_dataset.map(
        lambda example: {"label": [0 if example["difficulty"] == "hs" else 1]}
    )
    train_dataset = train_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(os.path.join(DIR, f"../models/{model_name}"))
