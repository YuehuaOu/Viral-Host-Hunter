import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = hidden_dims
        self._build_model()

    def _build_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.num_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.num_hidden[0], self.num_hidden[1]),
            nn.ReLU(),
            nn.Linear(self.num_hidden[1], self.num_hidden[2]),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden[2], self.num_hidden[1]),
            nn.ReLU(),
            nn.Linear(self.num_hidden[1], self.num_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.num_hidden[0], self.input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
