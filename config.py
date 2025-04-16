

from utils.tokenizer import tokenizer

class Config:
    # If you want to use MPS (Metal Performance Shaders) on macOS, uncomment the following line
    #device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = "cuda"
    lr = 1e-3
    num_epochs = 12
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    training_examples_size = 30000
    validation_examples_size = 500
    bidirectional_gru = True

    visualizer_example_sentences = [
        "I love this movie. It's amazing!",
        "This is the worst movie I've ever seen.",
        "The plot was boring and predictable.",
        "The acting was top-notch and the cinematography was stunning.",
        "I wouldn't recommend this film to anyone."
    ]

config = Config()