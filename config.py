
"""
Configuration Module

This module defines all configuration parameters for the sentiment classification model,
including model architecture, training settings, and example sentences for visualization.
"""

import torch  # Importing torch for device configuration
from utils.tokenizer import tokenizer

class Config:
    """
    Configuration class containing all parameters for the sentiment analysis model.
    
    This class centralizes all hyperparameters and settings to make experimentation
    and tuning easier. Parameters are organized into several categories:
    - Hardware settings (device)
    - Training hyperparameters (learning rate, epochs)
    - Model architecture (dimensions, layers)
    - Dataset configuration (sizes, tokenization)
    - Visualization settings (example sentences)
    """
    
    #--------------------- Hardware Configuration ---------------------#
    # Device for running the model (CPU, CUDA, MPS)
    # If you want to use MPS (Metal Performance Shaders) on macOS, uncomment the following line
    #device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = "cuda"  # Default to CUDA for GPU acceleration
    
    #--------------------- Training Hyperparameters ---------------------#
    lr = 1e-3         # Learning rate for the optimizer
    num_epochs = 12   # Number of training epochs
    
    #--------------------- Model Architecture ---------------------#
    embedding_dim = 100    # Dimension of word embeddings
    hidden_dim = 256       # Hidden dimension size for GRU units
    output_dim = 2         # Output dimension (binary: positive/negative)
    bidirectional_gru = True  # Whether to use bidirectional GRU (captures context from both directions)
    
    #--------------------- Tokenization Parameters ---------------------#
    # Using the BERT tokenizer settings
    pad_idx = tokenizer.pad_token_id  # Padding token ID for batch processing
    vocab_size = tokenizer.vocab_size  # Vocabulary size from the tokenizer
    
    #--------------------- Dataset Configuration ---------------------#
    # Subset sizes to use from the full SST-2 dataset
    training_examples_size = 30000    # Number of training examples to use
    validation_examples_size = 500    # Number of validation examples to use
    
    #--------------------- Visualization Settings ---------------------#
    # Example sentences with varying sentiment for visualization
    visualizer_example_sentences = [
        # Positive examples
        "I love this movie. It is amazing!",
        # Negative examples
        "This is the worst movie I have ever seen.",
        "The plot was boring and predictable.",
        # Positive examples with more complex structure
        "The acting was top-notch and the cinematography was stunning.",
        # Negative examples with different phrasing
        "I would not recommend this film to anyone."
    ]
    
    # Whether to generate visualizations after each epoch during training
    visualize_per_epoch = False

# Create a singleton instance of the configuration
config = Config()