import torch.nn as nn 
import torch.nn.init as init 


class NN(nn.Module):
    """
        Nearly the simplest feed-forward neural network you can imagine.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        hidden_layers=[64],
        activation=nn.ReLU, 
    ):
        super(NN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layer = nn.Linear(prev_dim, h)
            init.xavier_normal_(layer.weight)
            init.zeros_(layer.bias)

            layers.append(layer )
            layers.append(activation())

            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim) )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)