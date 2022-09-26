import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

class Decoder(nn.Module):
    """

    """
    def __init__(self, d_sequence, d_out, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_sequence = d_sequence
        self.d_out = d_out
        self.decoder = nn.Sequential(
            nn.Linear(d_out * d_sequence, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100)
        )

    def forward(self, X):
        X = X.reshape(-1, self.d_out*self.d_sequence)
        X = torch.sigmoid(self.decoder(X).reshape(-1, 10, 10))
        return X

class SI(nn.Module):
    """

    """
    def __init__(self, encoder_layer_num, pulse, embedding_dim, num_head, dropout, **kwargs):
        super(SI, self).__init__(**kwargs)
        self.encoder_layer_num = encoder_layer_num
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=40, nhead=num_head, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layer_num)
        self.decoder = Decoder(pulse, embedding_dim)

    def forward(self, X):
        X = self.encoder(X)
        # return torch.sigmoid(self.decoder(X))
        return torch.sigmoid(self.decoder(X))


"""Testing the shape of the inputs and outputs of the neural network architecture."""
in_data = torch.rand((126,20,20))
model = SI(4, 20, 20, 4, 0.1)
print(f"The dimention of input is {in_data.shape}")
print(f"The dimention of output is {model(in_data).shape}")
