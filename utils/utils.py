
import sys
import os
import glob

def make_checkpoint_name():
    """
    Generates a checkpoint name based on the current date and time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """
    Get the latest checkpoint file from the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to the latest checkpoint file
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        sys.exit(1)
        
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found in '{checkpoint_dir}'.")
        sys.exit(1)
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]