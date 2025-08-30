import yaml
import torch
from argparse import ArgumentParser
from src.data import load_tokenized_data
from src.train import create_model, train_model
from transformers import DataCollatorWithPadding


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", type="str", default="configs/training.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_config(args.config_file)

    tokenized_data, tokenizer = load_tokenized_data(config["dataset"], config["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    model = create_model(config["model_name"], config["num_labels"], device)
    train_model(model, tokenized_data, config, tokenizer, data_collator)


if __name__ == "__main__":
    main()
    