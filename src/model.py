import torch
from torch import nn


class RULLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("Model input size, hidden size, and layer count must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in the interval [0, 1).")
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
