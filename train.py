import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from src.data import tokenize, get_tokenized_data


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_and_prepare_data(dataset_name):
    dataset = load_dataset(dataset_name)
    tokenized_data = get_tokenized_data(dataset, tokenize)
    return tokenized_data
