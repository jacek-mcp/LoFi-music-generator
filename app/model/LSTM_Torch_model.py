import torch


class LSTMT_3embeddings(torch.nn.Module):

    def __init__(self, vocab_size, dur_vocab_size, velocities_vocab_size):

        super(LSTMT_3embeddings, self).__init__()
        hidden_dim = 512
        embbeding_dim = 64
        self.encoder = torch.nn.Embedding(vocab_size, embbeding_dim)
        self.encoder_dur = torch.nn.Embedding(dur_vocab_size, embbeding_dim)
        self.encoder_vel = torch.nn.Embedding(velocities_vocab_size, embbeding_dim)


        self.drop = torch.nn.Dropout(0.3)  # , 0.3, 0.5
        self.lstm = torch.nn.LSTM(embbeding_dim, hidden_dim,
                                  batch_first=True)  # try more lstm layers   # for bidirectional model check this https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py

        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
        self.decoder_dur = torch.nn.Linear(hidden_dim, dur_vocab_size)
        self.decoder_vel = torch.nn.Linear(hidden_dim, velocities_vocab_size)

        self.soft = torch.nn.LogSoftmax(dim=-1)


    def forward(self, x1, x2, x3, state=None):
        embeds = self.encoder(x1)
        embeds_dur = self.encoder_dur(x2)
        embeds_vel = self.encoder_vel(x3)

        merged_embeds = embeds + embeds_vel + embeds_dur
        drop = self.drop(merged_embeds)
        ht, state = self.lstm(drop, state)

        linear_out = self.decoder(ht)  # LSTM
        linear_out_vel = self.decoder_vel(ht)  # LSTM
        linear_out_dur = self.decoder_dur(ht)  # LSTM

        output = self.soft(linear_out)
        output_vel = self.soft(linear_out_vel)
        output_dur = self.soft(linear_out_dur)

        return output, output_dur, output_vel, state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


class LSTMT_chord(torch.nn.Module):

    def __init__(self, vocab_size):
        super(LSTMT_chord, self).__init__()
        hidden_dim = 512
        embbeding_dim = 64
        self.encoder = torch.nn.Embedding(vocab_size, embbeding_dim)

        self.drop = torch.nn.Dropout(0.3)  # , 0.3, 0.5
        self.lstm = torch.nn.LSTM(embbeding_dim, hidden_dim,
                                  batch_first=True)  # try more lstm layers   # for bidirectional model check this https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
        self.soft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, state=None):
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