from transformers import AutoTokenizer
from datasets import load_dataset


def load_tokenized_data(dataset_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset
