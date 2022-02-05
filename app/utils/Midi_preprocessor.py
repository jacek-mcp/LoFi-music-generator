import glob
import pickle
from fractions import Fraction

import numpy
import pandas as pd
import numpy as np
from music21 import converter, instrument, note, chord
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from google.colab import drive
from app.model.LSTM_Torch_model import LSTM

duration_list_full = []
velocity_list_full = []
position_list_full = []
notes_list_full = []
bins = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
centers = (bins[1:] + bins[:-1]) / 2

# for file in glob.glob("ff_midi/ff7tifa.mid"):
for file in glob.glob("drive/MyDrive/UPC-Project/blues/*.mid"):
    notes = []

    midi = converter.parse(file)

    print("Parsing %s" % file)

    notes_to_parse = None

    try:
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()

    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for position, element in enumerate(notes_to_parse[:]):
        if isinstance(element, note.Note):
            notes.append(
                [str(element.pitch), str(element.duration).split()[1].replace(">", ""), str(element.volume.velocity),
                 str(bins[np.digitize(round(position / len(notes_to_parse[:]), 1), centers)])])
        elif isinstance(element, chord.Chord):
            notes.append(
                ['.'.join(str(n) for n in element.normalOrder), str(element.duration).split()[1].replace(">", ""),
                 str(element.volume.velocity),
                 str(bins[np.digitize(round(position / len(notes_to_parse[:]), 1), centers)])])

    notes_list = [[item[0] for item in notes]]
    try:
        duration_list = [[float((Fraction(item[1]).numerator / Fraction(item[1]).denominator)) for item in notes]]
    except:
        try:
            duration_list = [[float(item[1]) for item in notes]]
        except:
            duration_list = [[item[1] for item in notes]]
    velocity_list = [[float(item[2]) for item in notes]]
    position_list = [[float(item[3]) for item in notes]]

    notes_list_full.append(notes_list)
    duration_list_full.append(duration_list)
    velocity_list_full.append(velocity_list)
    position_list_full.append(position_list)

    velocity_df = []
    for x in velocity_list_full:
        velocity_df.append(x[0])
    pd.DataFrame(velocity_df)
    vel_mean = []
    vel_min = []
    vel_max = []

    for col in pd.DataFrame(velocity_df).columns:
        vel_mean.append(pd.DataFrame(velocity_df)[col].mean())
        vel_min.append(pd.DataFrame(velocity_df)[col].min())
        vel_max.append(pd.DataFrame(velocity_df)[col].max())

    velocity_df_metrics = pd.DataFrame([vel_mean, vel_min, vel_max]).T
    velocity_df_metrics.columns = ["mean", "min", "max"]

velocity_df = []

for x in velocity_list_full:
    velocity_df.append(x[0])

pd.DataFrame(velocity_df)
vel_mean = []
vel_min = []
vel_max = []

for col in pd.DataFrame(velocity_df).columns:
    vel_mean.append(pd.DataFrame(velocity_df)[col].mean())
    vel_min.append(pd.DataFrame(velocity_df)[col].min())
    vel_max.append(pd.DataFrame(velocity_df)[col].max())

velocity_df_metrics = pd.DataFrame([vel_mean, vel_min, vel_max]).T
velocity_df_metrics.columns = ["mean", "min", "max"]

duration_df = []
for x in duration_list_full:
    duration_df.append(x[0])

dur_mean = []
dur_min = []
dur_max = []

for col in pd.DataFrame(duration_df).columns:
    try:
        dur_mean.append(pd.DataFrame(duration_df)[col].mean())
        dur_min.append(pd.DataFrame(duration_df)[col].min())
        dur_max.append(pd.DataFrame(duration_df)[col].max())
    except:
        pass

duration_df_metrics = pd.DataFrame([dur_mean, dur_min, dur_max]).T
duration_df_metrics.columns = ["mean", "min", "max"]

fig, ax = plt.subplots()
fig.set_size_inches(25, 15)

for i in ["mean"]:
    # ax.plot(foc_distro[foc_distro.order_discrepancy==f].day, foc_distro[
    # foc_distro.order_discrepancy==f].customers_share, label=f)
    ax.plot(duration_df_metrics[i], label=i)

ax.legend(loc='best')
plt.title("Duration Distribution along 100 Anime Songs")
plt.ylim(0, 4)
plt.xlabel("TimeStep")
plt.ylabel("Total Duration")
# pd.Series(vel_mean).plot()

fig, ax = plt.subplots()
fig.set_size_inches(25, 15)

for i in ["mean", "min", "max"]:
    # ax.plot(foc_distro[foc_distro.order_discrepancy==f].day, foc_distro[
    # foc_distro.order_discrepancy==f].customers_share, label=f)
    ax.plot(velocity_df_metrics[i], label=i)

ax.legend(loc='best')
plt.title("Velocity Distribution along 100 Anime Songs")
plt.xlabel("TimeStep")
plt.ylabel("Total Velocity")
# pd.Series(vel_mean).plot()

fig, ax = plt.subplots()
fig.set_size_inches(25, 15)
for i in range(0, 20):
    # ax.plot(foc_distro[foc_distro.order_discrepancy==f].day, foc_distro[
    # foc_distro.order_discrepancy==f].customers_share, label=f)
    ax.plot(pd.Series(velocity_list_full[i][0]), label=i)
ax.set_xlabel("timestep")
ax.set_ylabel("duration / velocity")
ax.legend(loc='best')
plt.xlim(0, 60)
plt.xticks(rotation=90)
plt.title("Velocity / Duration distribution")
plt.show()


def get_notes_v2():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    # positional bins
    bins = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    centers = (bins[1:] + bins[:-1]) / 2

    # for file in glob.glob("ff_midi/ff7tifa.mid"):
    for file in glob.glob("drive/MyDrive/UPC-Project/anime/anime/*.mid"):
        # for file in glob.glob("anime/anime/*.mid"):

        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            # for i in range(0,len(s2.parts)):
            # if s2.parts[i].partName == "Flute":
            # notes_to_parse = s2.parts[i].recurse()
            # print(notes_to_parse)
            notes_to_parse = s2.parts[0].recurse()

            # print(notes_to_parse)
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for position, element in enumerate(notes_to_parse[:]):

            if isinstance(element, note.Note):
                notes.append([str(element.pitch), str(element.duration).split()[1].replace(">", ""),
                              str(element.volume.velocity)])  # ,str(bins[np.digitize(round(position/len(notes_to_parse[:]),1) , centers)])
            if isinstance(element, chord.Chord):
                notes.append(
                    ['.'.join(str(n) for n in element.normalOrder), str(element.duration).split()[1].replace(">", ""),
                     str(element.volume.velocity), ])  # ,str(bins[np.digitize(round(position/len(notes_to_parse[:]),1) , centers)])
            if isinstance(element, note.Rest):
                notes.append(["Silence", str(element.duration).split()[1].replace(">", ""), str(0)])
                # else:
                # print("SILENCE____",element.seconds)

    with open('drive/MyDrive/UPC-Project/data/notes_big', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


def prepare_sequence(seq, char2idx, onehot=False):
    # convert sequence of words to indices
    idxs = [char2idx[c] for c in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    if onehot:
        # conver to onehot (if input to network)
        ohs = F.one_hot(idxs, len(char2idx)).long()
        return ohs
    else:
        return idxs


notes = get_notes_v2()

# Saving and storing Notes data
notes_list = [[item[0] for item in notes]]
with open('drive/MyDrive/UPC-Project/data/blues_notes', 'wb') as filepath:
    pickle.dump(notes_list, filepath)

velocity_list_dec = [[str(item[2]) for item in notes]]
velocity_list = [[]]
for x in velocity_list_dec[0]:
    if x != '0':
        velocity_list[0].append(x)
with open('drive/MyDrive/UPC-Project/data/blues_velocity', 'wb') as filepath:
    pickle.dump(velocity_list, filepath)

notes_duration_list = [["_".join([item[0], item[1]]) for item in notes]]
with open('drive/MyDrive/UPC-Project/data/blues_duration', 'wb') as filepath:
    pickle.dump(notes_duration_list, filepath)

duration_list = [[item[1] for item in
                  notes]]  # [[str((Fraction(item[1]).numerator/Fraction(item[1]).denominator)) for item in notes]]

# position_list = [[str(item[3]) for item in notes]]


drive.mount('/content/drive')


def note_formatting(notes_test):
    notes_formatted = []
    for y in notes_test:
        notes_formatted.append("_".join(y))
    return notes_formatted


notes_formatted = note_formatting(notes_list)
notes_formatted = notes_formatted[0].split("_")

notes_duration_formatted = notes_duration_list[0]

duration_formatted = note_formatting(duration_list)
duration_formatted = duration_formatted[0].split("_")

velocity_formatted = note_formatting(velocity_list)
velocity_formatted = velocity_formatted[0].split("_")

# get amount of pitch names
n_vocab_notes = len(set(notes_formatted))
n_vocab_duration = len(set(duration_formatted))
n_vocab_velocity = len(set(velocity_formatted))
n_vocab_notes_duration = len(set(notes_duration_formatted))



# merge the training data into one big text line
training_data = notes_list
#training_data = '_'.join(training_data[0])

# Assign a unique ID to each different character found in the training set
char2idx = {}
for sent in training_data:
    for c in sent:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
idx2char = dict((v, k) for k, v in char2idx.items())
VOCAB_SIZE = len(char2idx)
print('Number of found vocabulary tokens: ', VOCAB_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 500
SEQ_LEN= 32 #, 70, 80,

T = len(training_data[0])
CHUNK_SIZE = T // BATCH_SIZE
# let's first chunk the huge train sequence into BATCH_SIZE sub-sequences
trainset = [training_data[0][beg_i:end_i] \
            for beg_i, end_i in zip(range(0, T - CHUNK_SIZE, CHUNK_SIZE),
                                    range(CHUNK_SIZE, T, CHUNK_SIZE))]
print('Original training string len: ', T)
print('Sub-sequences len: ', CHUNK_SIZE)

training_data
train_set = []

# Create notes sequences of length 32.
for beg_t, end_t in zip(range(0, len(training_data[0]) -1 , SEQ_LEN + 1),
                          range(SEQ_LEN + 1, len(training_data[0]), SEQ_LEN + 1)):
    train_set.append(training_data[0][beg_t:end_t])


dataX = []
dataY = []

# Split input sequences and output sequences.
for seq in train_set:
    X = seq[:-1]
    Y = seq[1:]
    # convert each sequence to one-hots and labels respectively
    X = prepare_sequence(X, char2idx, onehot=False) #no onehot encoding.
    Y = prepare_sequence(Y, char2idx, onehot=False) #no onehot encoding.

    dataX.append(X.unsqueeze(0))
    dataY.append(Y.unsqueeze(0))
dataX = torch.cat(dataX, dim=0).to(device)
dataY = torch.cat(dataY, dim=0).to(device)


from torch.utils.data import Dataset, DataLoader

dataloaderX = DataLoader(dataX[:500], batch_size=10,shuffle=False) # batch_size=32,shuffle )
dataloaderY = DataLoader(dataY[:500], batch_size=10,shuffle=False )

batchX = next(iter(dataloaderX))
batchY = next(iter(dataloaderY))

print(batchX.shape)
print(dataX.shape)

word_embeddings = torch.nn.Embedding(n_vocab_notes, 64)
lstm = torch.nn.LSTM(64, 512, batch_first=True,bidirectional=False) #bidirectional=True ???
lin = torch.nn.Linear(512, n_vocab_notes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM(n_vocab_notes,n_vocab_notes).to(device)

#model.load_state_dict(torch.load("drive/MyDrive/UPC-Project/weigths/model.pt")) # blues

config = {
        "lr": 1e-3, #1e-3, # 1e-5 1e-3
        "batch_size": 64,

    }
optimizer = optim.Adam(model.parameters(), config["lr"])
criterion =F.nll_loss  #torch.nn.CrossEntropyLoss() #F.nll_loss
