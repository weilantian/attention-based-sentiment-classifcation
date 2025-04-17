import argparse
from datetime import datetime
from config import config
import torch

def setup_visualization_parser(subparsers):
    """
    Set up the parser for the visualization command.
    
    Args:
        subparsers: Subparsers object from argparse
    
    Returns:
        The configured parser
    """
    parser = subparsers.add_parser(
        'visualize',
        help='Visualize attention weights for sentiment analysis model'
    )

    parser.add_argument(
        "--sentences", 
        type=str, 
        nargs="+", 
        help="Sentences to analyze (wrap each in quotes)",
        default=config.visualizer_example_sentences
    )

    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to the checkpoint file (defaults to latest in ./checkpoints)",
        default=None
    )

    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path for the visualization",
        default=f"attention_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda","mps"], 
        help="Device to run the model on",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser



def setup_training_parser(subparsers):
    """
    Setup the argument parser for training.
    
    Args:
        subparsers: The subparsers object to add the training parser to.
    
    Returns:
        The configured training parser
    """
    
    parser = subparsers.add_parser(
        "train",
        help="Train the model"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=config.vocab_size,
        help="Size of the vocabulary"
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=config.embedding_dim,
        help="Dimension of the embedding layer"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=config.hidden_dim,
        help="Dimension of the hidden layer"
    )

    parser.add_argument(
        "--output_dim",
        type=int,
        default=config.output_dim,
        help="Dimension of the output layer"
    )

    parser.add_argument(
        "--pad_idx",
        type=int,
        default=config.pad_idx,
        help="Padding index"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda","mps"], 
        default=config.device,
        help="Device to use for training (cpu or cuda)"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.lr,
        help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=config.num_epochs,
        help="Number of epochs to train the model"
    )
    
    parser.add_argument(
        "--visualize_per_epoch",
        action="store_true",
        default=config.visualize_per_epoch,
        help="Generate attention visualizations after each training epoch"
    )

    return parser


def get_main_parser():
    """
    Create the main argument parser with all subcommands.
    
    Returns:
        The configured main parser
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis with Attention Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    setup_training_parser(subparsers)
    setup_visualization_parser(subparsers)
    return parser

def parse_args(args=None):
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    
    Returns:
        Parsed arguments
    """
    parser = get_main_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        exit(1)

    return parsed_args