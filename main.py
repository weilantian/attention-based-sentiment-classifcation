from utils import cli
import torch
from training.trainer import train

from utils.utils import get_latest_checkpoint
from visualization.attention_viz import visualize_multiple_examples
from models.model import SentimentClassificationModel
import sys
from utils.tokenizer import tokenizer
from config import config

#torch.set_num_threads(10)  # intra-op threads
#torch.set_num_interop_threads(2)  # usually smaller number is better here

def handle_visualization(args):

    model = SentimentClassificationModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        pad_idx=config.pad_idx,
        bidirectional=True,
    )

    """Handle the visualization command"""
    # Set device
    device = torch.device(args.device)
    
    # Get checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = get_latest_checkpoint()
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        sys.exit(1)
    
    # Visualize attention for the provided sentences
    fig = visualize_multiple_examples(
        model, 
        tokenizer, 
        args.sentences, 
        device,
        figsize=(15, 4*len(args.sentences)),
        output_file=args.output
    )

def handle_training(args):
    """Handle training cli commands"""
    device = torch.device(args.device)
    train(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        pad_idx=args.pad_idx,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )

def main():
    """Main entry point for the CLI"""

    args = cli.parse_args()

    if args.command == "train":
        handle_training(args)
    if args.command == "visualize":
        handle_visualization(args)

if __name__ == "__main__":
    main()