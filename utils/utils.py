
def make_checkpoint_name():
    """
    Generates a checkpoint name based on the current date and time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")