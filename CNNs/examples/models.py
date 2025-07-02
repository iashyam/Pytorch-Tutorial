import torch 
import torch.nn as nn


class fc(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_hidden_layers: int = 1,
                 hidden_dim: int = 64):
        
        """
        Fully connected neural network module.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_hidden_layers (int): Number of hidden layers in the network.
            hidden_dim (int): Dimension of each hidden layer.
        """
        super().__init__()

        self.sequence = nn.Sequential()
        self.sequence.add_module('input_layer', nn.Linear(input_dim, hidden_dim))
        self.sequence.add_module('input_activation', nn.ReLU())
        for i in range(num_hidden_layers - 1):
            self.sequence.add_module(f'hidden_layer_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.sequence.add_module(f'hidden_activation_{i}', nn.ReLU())
        self.sequence.add_module('output_layer', nn.Linear(hidden_dim, output_dim))        



    def forward(self, x):
        return self.sequence(x)
    


class VGG(nn.Module):
    def __init__(self, input_shape, output_dim: int):
        """
        VGG-like neural network module.
        Args:
            input_shape (tuple): Shape of the input features.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()

        C, H , W = input_shape
        min_channels = 64
        max_channels = 128
        fc_dim = 512
        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(C, min_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(min_channels, min_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(min_channels, max_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(max_channels, max_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #fully connected layers
        flattend_dim = (H // 4) * (W // 4) * max_channels
        self.sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattend_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim)
        )
   
    def forward(self, x):
        x = self.convs(x)
        x = self.sequence(x)
        return x
