import yaml
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.data import load_tokenized_data


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
