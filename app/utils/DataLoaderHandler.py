import torch
from torch.utils.data import Dataset, DataLoader

from app.utils.Midi_preprocessor import MidiPreprocessor


class DataLoaderHandler:

    def __init__(self, path):
        self.preprocessor = MidiPreprocessor(path)

    def get_note_velocity_loaders(self, data):

        BATCH_SIZE = 135
        SEQ_LEN = 32  # , 70, 80,

        train_set = []

        # Create notes sequences of length 32.
        for beg_t, end_t in zip(range(0, len(data) - 1, SEQ_LEN + 1),
                                range(SEQ_LEN + 1, len(data), SEQ_LEN + 1)):
            train_set.append(data[beg_t:end_t])

        dataX = []
        dataY = []
        # Split input sequences and output sequences.

        vel2idx, idx2vel, vel_VOCAB_SIZE = self.preprocessor.load_velocity_vocab_dicts()
        char2idx_note_dur, idx2char_note_dur, note_dur_VOCAB_SIZE = self.preprocessor.load_notes_durations_vocab_dicts()

        for seq in train_set:
            X = seq[:-1]
            Y = seq[1:]
            X = self.preprocessor.prepare_sequence(X, vel2idx, char2idx_note_dur)  # no onehot encoding.
            Y = self.preprocessor.prepare_sequence(Y, vel2idx, char2idx_note_dur)  # no onehot encoding.

            dataX.append(torch.stack(X))
            dataY.append(torch.stack(Y))

        dataloaderX = DataLoader(dataX[:1620], batch_size=BATCH_SIZE, shuffle=False)  # batch_size=32,shuffle )
        dataloaderY = DataLoader(dataY[:1620], batch_size=BATCH_SIZE, shuffle=False)
        return dataloaderX, dataloaderY

    def get_chord_loaders(self, data):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 5
        SEQ_LEN = 4  # , 70, 80,

        train_set = []

        # Create notes sequences of length 32.
        for beg_t, end_t in zip(range(0, len(data) - 1, SEQ_LEN + 1),
                                range(SEQ_LEN + 1, len(data), SEQ_LEN + 1)):
            train_set.append(data[beg_t:end_t])

        dataX = []
        dataY = []

        chord2idx, idx2chord, chord_VOCAB_SIZE = self.preprocessor.load_chords_vocab_dicts()
        # Split input sequences and output sequences.
        for seq in train_set:
            X = seq[:-1]
            Y = seq[1:]
            X = self.preprocessor.prepare_sequence_chords(X, chord2idx, onehot=False)  # no onehot encoding.
            Y = self.preprocessor.prepare_sequence_chords(Y, chord2idx, onehot=False)  # no onehot encoding.

            dataX.append(X.unsqueeze(0))
            dataY.append(Y.unsqueeze(0))

            # dataY.append(Y)
        dataX = torch.cat(dataX, dim=0).to(device)
        dataY = torch.cat(dataY, dim=0).to(device)

        dataloaderX = DataLoader(dataX[:], batch_size=BATCH_SIZE, shuffle=False)  # batch_size=32,shuffle )
        dataloaderY = DataLoader(dataY[:], batch_size=BATCH_SIZE, shuffle=False)

        return dataloaderX, dataloaderY

