from datasets import load_dataset
from utils.tokenizer import preprocess_func
import numpy as np
from torch.utils.data import Subset
from config import config
# The SST-2 dataset is a sentiment analysis dataset from the GLUE benchmark
# It contains 67,349 training examples and 872 examples for validation.
# The dataset is split into two classes: positive and negative.
sst2_dataset = load_dataset("glue", "sst2")

# 1. Preprocess the dataset
tokenized_dataset = sst2_dataset.map(preprocess_func, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 2. Create subsets for faster training
train_indices = np.random.choice(len(tokenized_dataset["train"]), size=config.training_examples_size, replace=False)
train_subset = Subset(tokenized_dataset["train"], train_indices)

val_indices = np.random.choice(len(tokenized_dataset["validation"]), size=config.validation_examples_size, replace=False)
val_subset = Subset(tokenized_dataset["validation"], val_indices)