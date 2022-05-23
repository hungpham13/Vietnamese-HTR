import torch
from torch import nn


class BRNN(nn.Module):
    def __init__(self, input_size, vocab_size):
        super(BRNN, self).__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size=512, dropout=0.2, bidirectional=True,
                               batch_first=True, num_layers=2)
        # self.bilstm2 = nn.LSTM(512, 512, dropout=0.2, bidirectional=True,
        #                        batch_first=True)
        self.fc = nn.Sequential(nn.Linear(512, vocab_size + 1),
                                nn.Softmax(vocab_size + 1)
                                )

	def _init_hidden(self, x: torch.Tensor):
		return torch.zeros((4, x.shape[0], 512)), 
			   torch.zeros((4, x.shape[0], 512))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (N,L,H): Batch size, Length seq, Channels
        :return: shape (240, vocab_size + 1)
        """
        hidden = self._init_hidden(x)
        x, hidden = self.bilstm1(x, hidden)
        print(x.shape)
        x = self.fc(x)
        return x
