import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
