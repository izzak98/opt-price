"""LSTM model for quantile regression."""
import json
import torch
import torch.nn as nn
from torch import Tensor

# from utils.train_utils import TwoStageQuantileLoss as FullQuantileLoss
# from utils.train_utils import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


class LSTM_Model(nn.Module):
    """
    LSTM model for quantile regression.

    Parameters:
        lstm_layers (int): Number of LSTM layers for the normalization module.
        lstm_h (int): Number of hidden units in the LSTM layers.
        hidden_layers (list[int]): Number of units in each hidden layer for the
        normalization module.
        hidden_activation (str): Activation function for hidden layers in the normalization module.
        market_lstm_layers (int): Number of LSTM layers for the market module.
        market_lstm_h (int): Number of hidden units in the LSTM layers for the market module.
        market_hidden_layers (list[int]): Number of units in each hidden layer
        for the market module.
        market_hidden_activation (str): Activation function for hidden layers in the market module.
        dropout (float): Dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        input_size (int): Number of features in the raw data.
        market_data_size (int): Number of features in the market data.

    Inputs:
        x (torch.Tensor): Input tensor of raw data.
        s (torch.Tensor): Input tensor of standardized data.
        z (torch.Tensor): Input tensor of market data.

    Returns:
        normalized_output (torch.Tensor): Normalized output tensor.
        raw_output (torch.Tensor): Raw output tensor.
    """

    def __init__(self,
                 lstm_layers: int,
                 lstm_h: int,
                 hidden_layers: list[int],
                 hidden_activation: str,
                 market_lstm_layers: int,
                 market_lstm_h: int,
                 market_hidden_layers: list[int],
                 market_hidden_activation: str,
                 dropout: float,
                 layer_norm: bool = False,
                 input_size: int = 49,
                 market_data_size: int = 21,
                 output_size: int = 37
                 ) -> None:
        super().__init__()
        self.layer_norm = layer_norm
        self.dropout = dropout

        # Normalize module LSTM and Linear layers
        self.normalize_lstm = nn.LSTM(
            input_size, lstm_h, lstm_layers, dropout=dropout, batch_first=True)
        self.normalize_module = self._build_module(
            lstm_h, hidden_layers, output_size, hidden_activation)

        # Market module LSTM and Linear layers
        self.market_lstm = nn.LSTM(
            market_data_size, market_lstm_h, market_lstm_layers, dropout=dropout, batch_first=True)
        self.market_module = self._build_module(
            market_lstm_h, market_hidden_layers, 1, market_hidden_activation)

    def _build_module(self, input_size, hidden_layers, output_size, activation) -> nn.Sequential:
        layers = []
        for i, neurons in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size if i ==
                          0 else hidden_layers[i-1], neurons))
            if self.layer_norm:
                layers.append(nn.LayerNorm(neurons))
            if activation:
                layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        if activation not in activations:
            raise ValueError(f"Activation {activation} not supported")
        return activations[activation]

    def forward(self, x, s, z) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        # x: (batch_size, seq_len, input_size)
        # s: (batch_size, seq_len, 1)
        # z: (batch_size, seq_len, market_data_size)

        # Normalize LSTM stage
        normalized_lstm_output, _ = self.normalize_lstm(x)
        normalized_lstm_output = normalized_lstm_output[:, -1, :]
        # (batch_size,  output_size)
        normalized_output = self.normalize_module(normalized_lstm_output)
        normalized_output, _ = torch.sort(normalized_output, dim=1)

        # Apply scaling factor from 's'
        new_output = normalized_output  # * s

        # Market LSTM stage
        market_lstm_output, _ = self.market_lstm(z)
        market_lstm_output = market_lstm_output[:, -1, :]
        # (batch_size, seq_len, 1)
        estimated_sigma = self.market_module(market_lstm_output)

        # Raw output scaling with estimated sigma
        raw_output = new_output * estimated_sigma  # Element-wise multiplication

        return normalized_output, raw_output
