from torch import nn

# The attention model that captures the importance of different parts of the input
class AttentionModel(nn.Module):
    """
    A additive-attention model for sentiment classification.
    """