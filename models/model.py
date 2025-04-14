from torch import nn

class SentimentClassificationModel(nn.Module):
    """
    A simple sentiment classification model using a feedforward neural network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x