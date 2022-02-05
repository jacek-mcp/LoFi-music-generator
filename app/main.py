import torch
import app.model.LSTM_Torch_model as LSTMT
import app.model.LSTMT_2embeddings as LSTMT_2embeddings
from timeit import default_timer as timer
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

config = {
    "lr": 1e-3,  # 1e-3, # 1e-5 1e-3
    "batch_size": 64,

}


def get_data_loaders():
    data = [notes_duration_formated_v2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 500
    SEQ_LEN = 32  # , 70, 80,

    T = len(data[0])  # len(training_data[0])
    CHUNK_SIZE = T // BATCH_SIZE
    # let's first chunk the huge train sequence into BATCH_SIZE sub-sequences
    trainset = [data[0][beg_i:end_i]
                for beg_i, end_i in zip(range(0, T - CHUNK_SIZE, CHUNK_SIZE),
                                        range(CHUNK_SIZE, T, CHUNK_SIZE))]
    print('Original training string len: ', T)
    print('Sub-sequences len: ', CHUNK_SIZE)

    training_data
    train_set = []

    # Create notes sequences of length 32.
    for beg_t, end_t in zip(range(0, len(data[0]) - 1, SEQ_LEN + 1),
                            range(SEQ_LEN + 1, len(data[0]), SEQ_LEN + 1)):
        train_set.append(data[0][beg_t:end_t])

    dataX = []
    dataY = []

    # Split input sequences and output sequences.
    for seq in train_set:
        X = seq[:-1]
        Y = seq[1:]
        # convert each sequence to one-hots and labels respectively
        X = prepare_sequence_old(X, char2idx, onehot=False)  # no onehot encoding.
        Y = prepare_sequence_old(Y, char2idx, onehot=False)  # no onehot encoding.

        dataX.append(X.unsqueeze(0))
        dataY.append(Y.unsqueeze(0))
    dataX = torch.cat(dataX, dim=0).to(device)
    dataY = torch.cat(dataY, dim=0).to(device)

    dataloaderX = DataLoader(dataX[:], batch_size=137, shuffle=False)  # batch_size=32,shuffle )
    dataloaderY = DataLoader(dataY[:], batch_size=137, shuffle=False)

    batchX = next(iter(dataloaderX))
    batchY = next(iter(dataloaderY))


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMT(note_dur_VOCAB_SIZE).to(device)
    # model_velocity = LSTMT(note_vel_VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("drive/MyDrive/UPC-Project/weights/test_0.6.pt"))  # blues

    model_two_embeddings = LSTMT_2embeddings(note_dur_VOCAB_SIZE, note_vel_VOCAB_SIZE).to(device)
    # model_two_embeddings_2 = LSTMT_2embeddings_v2(note_dur_VOCAB_SIZE,vel_VOCAB_SIZE,note_dur_vel_VOCAB_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), config["lr"])
    criterion = torch.nn.CrossEntropyLoss()  # F.nll_loss  #torch.nn.CrossEntropyLoss() #F.nll_loss  /  label smoothing pytorch

    # def repackage_hidden(h):
    #    """Wraps hidden states in new Tensors, to detach them from their history."""
    #    if isinstance(h, torch.Tensor):
    #        return h.detach()
    #    else:
    #        return tuple(repackage_hidden(v) for v in h)  # For LSTMs

    # Let's now build a model to train with its optimizer and loss
    model.to(device)
    NUM_EPOCHS = 100
    tr_loss = []
    avg_loss = None
    avg_weight = 0.1
    state = None

    timer_beg = timer()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in zip(dataloaderX, dataloaderY):

            # x shape is [batch_size,2,sequence_length] where 2 is the sequence for notes_duration and velocity

            # get notes and durations index sequence
            notes_dur = [x[0] for x in x]
            x1 = torch.stack(notes_dur)

            # get velocities index sequence
            velocities = [x[1] for x in x]
            x2 = torch.stack(velocities)

            # same for target - Y
            notesY = [x[0] for x in y]
            y1 = torch.stack(notesY)
            velocitiesY = [x[1] for x in y]
            y2 = torch.stack(velocitiesY)

            optimizer.zero_grad()
            # Step 3. Run our forward pass. get two outputs: (notes_durations + velocity)
            y_1, y_2, state = model_two_embeddings(x1, x2, state)

            # detach the previous state graph to not backprop gradients further than the BPTT span
            state = (state[0].detach(),  # detach c[t]
                     state[1].detach())  # detach h[t]

            # Predict
            y_1 = y_1.reshape(-1, note_dur_VOCAB_SIZE)  # >> only when using batches
            loss1 = F.nll_loss(y_1, y1.view(-1))

            y_2 = y_2.reshape(-1, note_vel_VOCAB_SIZE)  # >> only when using batches
            loss2 = F.nll_loss(y_2, y2.view(-1))

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            if avg_loss:
                avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
            else:
                avg_loss = loss.item()
            tr_loss.append(loss.item())

        timer_end = timer()
        PATH = "drive/MyDrive/UPC-Project/weights/model_two_embeddings{}.pt".format(round(float(loss), 1))
        torch.save(model.state_dict(), PATH)
        if (epoch + 1) % 1 == 0:
            # Generate a seed sentence to play around
            model.to('cpu')
            print('-' * 30)
            print('-' * 30)
            model.to(device)
            print('Finished epoch {} in {:.1f} s: loss: {:.6f}'.format(epoch + 1,
                                                                       timer_end - timer_beg,
                                                                       np.mean(tr_loss[-10:])))
        timer_beg = timer()

    plt.plot(tr_loss)
    plt.xlabel('Epoch')
    plt.ylabel('NLLLoss')


if __name__ == '__main__':
    print("dupsko")
