from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
from typing import Tuple
from src.types import DatasetType 


def load_tokenized_data(dataset_name: str, 
                        model_name: str
                        ) -> Tuple[DatasetType, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset, tokenizer
