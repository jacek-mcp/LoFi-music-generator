import torch


class LSTMT_2embeddings(torch.nn.Module):

    def __init__(self, notes_vocab_size, velocities_vocab_size):
        super(LSTMT_2embeddings, self).__init__()
        hidden_dim = 512
        embbeding_dim = 64
        self.encoder_notes = torch.nn.Embedding(notes_vocab_size, embbeding_dim)
        self.encoder_vel = torch.nn.Embedding(velocities_vocab_size, embbeding_dim)

        self.drop = torch.nn.Dropout(0.3)
        self.lstm = torch.nn.LSTM(embbeding_dim, hidden_dim, batch_first=True)

        # two different linear layers for each output (notes_duration and Velocity)
        self.decoder = torch.nn.Linear(hidden_dim, notes_vocab_size)
        self.decoder_vel = torch.nn.Linear(hidden_dim, velocities_vocab_size)

        self.soft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x_notes, x_vel, state=None):
        embeds = self.encoder_notes(x_notes)
        embeds_vel = self.encoder_vel(x_vel)

        # sum the embeddings layers
        merged_emb = embeds + embeds_vel

        ht, state = self.lstm(merged_emb, state)

        # linear outputs with different vocab sizes
        linear_out = self.decoder(ht)
        linear_out_vel = self.decoder_vel(ht)

        output = self.soft(linear_out)
        output_vel = self.soft(linear_out_vel)

        return output, output_vel, state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))