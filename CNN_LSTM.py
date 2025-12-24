# CNN_LSTM.py

import torch
import torch.nn as nn

# DEFINE CNN LSTM MODEL
class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        cnn_out_channels: int,
        cnn_kernel_size: int,
        pooling_kernel_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ):
        """
        CNN-LSTM model for sequence-to-vector forecasting.

        Args:
            input_size: Number of features per time step.
            cnn_out_channels: Number of filters in the convolutional layer.
            cnn_kernel_size: Size of the convolutional kernel.
            pooling_kernel_size: Size of the pooling kernel (reduces sequence length).
            hidden_size: Number of features in the LSTM hidden state.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate (applied if num_layers > 1).
            output_size: Number of output predictions (forecast horizon).
        """
        super().__init__()

        # CNN expects input: (batch, channels, sequence_length)
        self.cnn = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,  # keep length roughly the same pre-pooling
        )
        self.relu = nn.ReLU()

        # Pool along time axis: reduces sequence length
        self.pool = nn.MaxPool1d(kernel_size=pooling_kernel_size)

        # LSTM expects input: (batch, seq_len, features) when batch_first=True
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)

        x = self.cnn(x)
        x = self.relu(x)
        x = self.pool(x)

        # (batch, cnn_out_channels, new_seq_len) -> (batch, new_seq_len, cnn_out_channels)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        # Take the last timestep output
        out = self.fc(lstm_out[:, -1, :])
        return out