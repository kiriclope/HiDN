class Perceptron(nn.Module):
    def __init__(self, num_features, dropout_rate=0.0):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        hidden = self.linear(x)
        return hidden

class MLP(nn.Module):
    def __init__(self, num_features, hidden_units=64, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.linear = nn.Linear(num_features, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        hidden = self.linear2(x)
        return hidden
