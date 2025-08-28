def tokenize(dataset, tokenizer):
    return tokenizer(dataset["text"], truncation=True)

def get_tokenized_data(dataset, tokenize_function=tokenize):
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
