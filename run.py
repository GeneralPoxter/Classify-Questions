"""
Developer: Jason Liu
Comments: Testing and training script for classifiers
"""

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    AlbertForSequenceClassification,
    AlbertTokenizerFast,
    ConvBertForSequenceClassification,
    ConvBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    ElectraForSequenceClassification,
    ElectraTokenizerFast,
)
from classify.train import train
from classify.test import test
import argparse
import numpy as np
import os

DIR = os.path.dirname(__file__)

# TODO: add more viable classifiers
classifiers = {
    "bert": {
        "model": BertForSequenceClassification,
        "tokenizer": BertTokenizerFast,
        "src": "bert-base-cased",
    },
    "albert": {
        "model": AlbertForSequenceClassification,
        "tokenizer": AlbertTokenizerFast,
        "src": "albert-base-v2",
    },
    "distil-bert": {
        "model": DistilBertForSequenceClassification,
        "tokenizer": DistilBertTokenizerFast,
        "src": "distilbert-base-uncased",
    },
    "conv-bert": {
        "model": ConvBertForSequenceClassification,
        "tokenizer": ConvBertTokenizerFast,
        "src": "YituTech/conv-bert-base",
    },
    "electra": {
        "model": ElectraForSequenceClassification,
        "tokenizer": ElectraTokenizerFast,
        "src": "google/electra-small-discriminator",
    },
}

parser = argparse.ArgumentParser(
    description="Train and test classifiers on question dataset"
)
parser.add_argument(
    "classifiers",
    type=str,
    nargs="+",
    help="names of classifiers to train and test",
)
parser.add_argument(
    "--test-only",
    action="store_true",
    help="only test the pretrained classifiers",
)

args = parser.parse_args()
classifier_names = np.unique(args.classifiers)

for classifier_name in classifier_names:
    if classifier_name in classifiers:
        classifier = classifiers[classifier_name]
        tokenizer = classifier["tokenizer"].from_pretrained(classifier["src"])

        if not args.test_only:
            print(f"Training {classifier_name}...")
            model = classifier["model"].from_pretrained(classifier["src"])
            train(model, classifier_name, tokenizer)

        model_path = os.path.join(DIR, f"models/{classifier_name}")
        if os.path.isdir(model_path):
            print(f"Testing {classifier_name}...")
            model = classifier["model"].from_pretrained(model_path)
            test(model, classifier_name, tokenizer)
        else:
            print(f"{model_path} not found, aborting {classifier_name} model test...")
    else:
        print(f"{classifier_name} not recognized, ignoring...")
