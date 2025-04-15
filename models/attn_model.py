import torch
from torch import nn

# The attention model that captures the importance of different parts of the input
class AttentionModel(nn.Module):
    """
    A additive-attention model for sentiment classification.
    """
    def __init__(self,gru_output_dim,gru_hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(gru_output_dim,gru_hidden_dim)
        self.Ua = nn.Linear(gru_hidden_dim,gru_hidden_dim)
        self.Va = nn.Linear(gru_hidden_dim,1)



    def forward(self,gru_output,hidden_state):
        energy = torch.tanh(self.Wa(gru_output) + self.Ua(hidden_state).unsqueeze(1))
        attn_weights = self.Va(energy).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, gru_output).squeeze(1)
        return context, attn_weights
