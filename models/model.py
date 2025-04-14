from torch import nn
import torch
from attn_model import AttentionModel

class SentimentClassificationModel(nn.Module):
    """
    A simple sentiment classification model using a feedforward neural network.
    """

    def __init__(self,vocab_size, embedding_dim, hidden_dim, output_dim,pad_idx,n_layers=1, bidirectional=False, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim,
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            batch_first=True, 
            dropout=0 if n_layers<2 else dropout)
        self.gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.gru_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.attention = AttentionModel(self.gru_output_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

    def forward(self, text, return_attention=False):
        batch_size = text.shape[0]
        embedded = self.embedding(text)

        h_0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim).to(embedded.device)

        gru_output, hidden = self.gru(embedded, h_0)
        
        # Get the final hidden state
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]  # [batch_size, hidden_dim]

        context, attn_weights = self.attention(gru_output, hidden)

        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if isinstance(return_attention, bool) and return_attention:
            return x, attn_weights
        
        return x