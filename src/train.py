from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    PreTrainedTokenizerBase
)
from src.evaluate import compute_metrics
from typing import Dict, Any
from datasets import DatasetDict


def create_model(model_name: str, num_labels: int, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to(device)
    return model

def create_training_args(config: Dict[str, Any]):
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
        logging_dir=f"{config["output_dir"]}/logs",
        logging_steps=10
    )
    return training_args

def train_model(model, 
                tokenized_data: DatasetDict, 
                config: Dict[str, Any], 
                tokenizer: PreTrainedTokenizerBase, 
                data_collator):
    training_args = create_training_args(config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"], 
        eval_dataset=tokenized_data["validation"], 
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics # type: ignore
    )
    
    trainer.train()
    trainer.save_model(output_dir=config["output_dir"])
    return trainer
