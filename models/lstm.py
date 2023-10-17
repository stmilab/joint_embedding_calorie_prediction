import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self, input_size=2, hidden_size=64, output_size=1, dropout=0, device="cpu"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=self.num_layers, dropout=dropout
        )

        self._dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self._sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, input_seq, train=False):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(
            self.device
        )

        input_seq = input_seq.transpose(0, 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq, (h0, c0))
        if train:
            lstm_out = self._dropout(lstm_out)

        # return self.softmax(predictions)
        # predictions = self._sigmoid(predictions)

        return lstm_out[-1]
