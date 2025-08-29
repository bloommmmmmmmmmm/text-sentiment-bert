import yaml
import torch
from argparse import ArgumentParser
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from src.data import load_tokenized_data
from src.evaluate import compute_metrics


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, data, config, tokenizer, data_collator):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10
    )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"], # type: ignore
        eval_dataset=data["validation"], # type: ignore
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics # type: ignore
    )
    
    trainer.train()
    trainer.save_model(output_dir=config["output_dir"])
    return trainer

def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", type="str", default="configs/training.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_config(args.config_file)

    tokenized_data, tokenizer = load_tokenized_data(config["dataset"], config["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=config["num_labels"]
    ).to(device)
    train_model(model=model, 
                data=tokenized_data,
                config=config, 
                tokenizer=tokenizer,
                data_collator=data_collator
    )


if __name__ == "__main__":
    main()
