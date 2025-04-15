import time
from datasets import load_dataset
from utils.tokenizer import preprocess_func
import numpy as np
from torch.utils.data import Subset
from config import config

print("Starting dataset loading...")
start_time = time.time()

# The SST-2 dataset is a sentiment analysis dataset from the GLUE benchmark
sst2_dataset = load_dataset("glue", "sst2")
dataset_load_time = time.time() - start_time
print(f"Dataset loading time: {dataset_load_time:.2f} seconds")

# 1. Preprocess the dataset
print("Starting tokenization...")
tokenize_start_time = time.time()
tokenized_dataset = sst2_dataset.map(preprocess_func, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenize_time = time.time() - tokenize_start_time
print(f"Tokenization time: {tokenize_time:.2f} seconds")

# 2. Create subsets for faster training
print("Creating subsets...")
subset_start_time = time.time()
train_indices = np.random.choice(len(tokenized_dataset["train"]), size=config.training_examples_size, replace=False)
train_subset = Subset(tokenized_dataset["train"], train_indices)

val_indices = np.random.choice(len(tokenized_dataset["validation"]), size=config.validation_examples_size, replace=False)
val_subset = Subset(tokenized_dataset["validation"], val_indices)
subset_time = time.time() - subset_start_time
print(f"Subset creation time: {subset_time:.2f} seconds")

total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")