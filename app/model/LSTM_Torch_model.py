import torch


class LSTM(torch.nn.Module):

    def __init__(self, vocab_size):
        super(LSTM, self).__init__()
        hidden_dim = 512
        embbeding_dim = 64
        self.encoder = torch.nn.Embedding(vocab_size, embbeding_dim)

        self.drop = torch.nn.Dropout(0.3)  # , 0.3, 0.5
        self.lstm = torch.nn.LSTM(embbeding_dim, hidden_dim,
                                  batch_first=True)  # try more lstm layers   # for bidirectional model check this https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
        self.gru = torch.nn.GRU(embbeding_dim, hidden_dim,
                                batch_first=True)  # try more lstm layers   # for bidirectional model check this https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py

        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
        self.soft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, state=None):
        # batch size = 1
        embeds = self.encoder(x)
        drop = self.drop(embeds)
        ht, state = self.lstm(drop, state)

        linear_out = self.decoder(ht)  # LSTM

        output = self.soft(linear_out)
        return output, state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
