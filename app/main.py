import numpy as np
import torch
import app.model.LSTM_Torch_model as LSTMT
import torch.optim as optim
import torch.nn.functional as F
from timeit import default_timer as timer
from music21 import instrument, note, stream, chord, meter, tempo, audioSearch, common, corpus, pitch

from app.utils.DataLoaderHandler import DataLoaderHandler
from app.utils.DataManager import DataManager
from app.utils.MidiToWav import MidiToWav
from app.utils.Midi_preprocessor import MidiPreprocessor
from app.utils.mappers.EffectsMapper import EffectsMapper
from app.utils.mappers.InstrumentsMapper import InstrumentsMapper

import numpy
from music21 import instrument, note, stream, chord
import music21
from fractions import Fraction
import os
import time
import sys

config = {
    "lr": 1e-3,  # 1e-3, # 1e-5 1e-3
    "batch_size": 64,

}

PATH = os.getcwd() + '/app/data/'
MODELS_PATH = PATH + '/models/'
MIDI_PATH = PATH + '/midi/'
GEN_SIZE = 200
TEMPO = 90


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def gen_notes_three_embeddings(model_three_embeddings, num_chars):
    data_manager = DataManager()

    valocity_vocab_dict = data_manager.load_velocity_vocab_dicts()
    duration_vocab_dict = data_manager.load_duration_vocab_dicts()
    notes_vocab_dict = data_manager.load_notes_vocab_dicts()
    duration_velocity_vocab_dict = data_manager.load_notes_duration_velocity_vocab_dicts()

    list_of_first_notes = None
    random_notes = np.random.randint(0, len([x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()]))
    list_of_first_notes = [[x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()][random_notes]]

    seed = list_of_first_notes
    preds = None

    model_three_embeddings.eval()
    state = None

    inputs = MidiPreprocessor.prepare_sequence(seed, notes_vocab_dict['char2idx'], duration_vocab_dict['dur2idx'],
                                               valocity_vocab_dict['vel2idx'])

    x1 = inputs[0].reshape(1, len(inputs[0]))
    x2 = inputs[1].reshape(1, len(inputs[1]))
    x3 = inputs[2].reshape(1, len(inputs[2]))

    seed_pred, dur_pred, vel_pred, state = model_three_embeddings(x1, x2, x3,state)
    seed_pred = seed_pred.reshape(-1, notes_vocab_dict['VOCAB_SIZE'])
    dur_pred = dur_pred.reshape(-1, duration_vocab_dict['dur_VOCAB_SIZE'])
    vel_pred = vel_pred.reshape(-1, valocity_vocab_dict['vel_VOCAB_SIZE'])

    preds = seed
    curr_pred = torch.topk(seed_pred[-1, :], k=1, dim=0)[1]
    curr_pred_dur = torch.topk(dur_pred[-1, :], k=1, dim=0)[1]
    curr_pred_vel = torch.topk(vel_pred[-1, :], k=1, dim=0)[1]

    curr_pred = notes_vocab_dict['idx2char'][curr_pred.item()]
    curr_pred_dur = duration_vocab_dict['idx2dur'][curr_pred_dur.item()]
    curr_pred_vel = valocity_vocab_dict['idx2vel'][curr_pred_vel.item()]

    preds.append("_".join([curr_pred, curr_pred_dur, curr_pred_vel]))

    for t in range(num_chars):
        if len(intersection(preds[-4:], preds[-8:-4])) / len(preds[-4:]) >= 0.75:
            # print("random1")
            random_note = np.random.randint(0, len([x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()]))
            curr_pred = [x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()][random_note]  # or always same random_notes
        elif len(intersection(preds[-8:], preds[-16:-8])) / len(preds[-8:]) >= 0.75:
            # print("random2")
            random_note = np.random.randint(0, len([x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()]))
            curr_pred = [x for x in duration_velocity_vocab_dict['char2idx_note_dur_vel'].keys()][random_note]  # or always same random_notes
        else:

            curr_pred = "_".join([curr_pred, curr_pred_dur, curr_pred_vel])

        curr_pred = MidiPreprocessor.prepare_sequence([curr_pred], notes_vocab_dict['char2idx'], duration_vocab_dict['dur2idx'],
                                          valocity_vocab_dict['vel2idx'])

        curr_pred_note = curr_pred[0].reshape(1, len(curr_pred[0]))
        curr_pred_dur = curr_pred[1].reshape(1, len(curr_pred[1]))
        curr_pred_vel = curr_pred[2].reshape(1, len(curr_pred[2]))

        curr_pred, curr_pred_dur, curr_pred_vel, state = model_three_embeddings(curr_pred_note, curr_pred_dur,
                                                                                curr_pred_vel, state)

        curr_pred = curr_pred.reshape(-1, notes_vocab_dict['VOCAB_SIZE'])
        curr_pred_dur = curr_pred_dur.reshape(-1, duration_vocab_dict['dur_VOCAB_SIZE'])
        curr_pred_vel = curr_pred_vel.reshape(-1, valocity_vocab_dict['vel_VOCAB_SIZE'])

        curr_pred = torch.topk(curr_pred[-1, :], k=1, dim=0)[1]
        curr_pred_dur = torch.topk(curr_pred_dur[-1, :], k=1, dim=0)[1]
        curr_pred_vel = torch.topk(curr_pred_vel[-1, :], k=1, dim=0)[1]

        curr_pred = notes_vocab_dict['idx2char'][curr_pred.item()]
        curr_pred_dur = duration_vocab_dict['idx2dur'][curr_pred_dur.item()]
        curr_pred_vel = valocity_vocab_dict['idx2vel'][curr_pred_vel.item()]

        preds.append("_".join([curr_pred, curr_pred_dur, curr_pred_vel]))

    return preds


def gen_notes_chords(model, num_chars):
    data_manager = DataManager()

    chords_vocab_dict = data_manager.load_chords_vocab_dicts()

    model.eval()
    state = None

    random_notes = np.random.randint(0, len([x for x in chords_vocab_dict['chord2idx'].keys()]))
    seed = [[x for x in chords_vocab_dict['chord2idx'].keys()][random_notes]]

    inputs = MidiPreprocessor.prepare_sequence_chords(seed, chords_vocab_dict['chord2idx'], onehot=False)
    inputs = inputs.reshape(1, len(inputs))
    seed_pred, state = model(inputs, state)
    seed_pred = seed_pred.reshape(-1, chords_vocab_dict['chord_VOCAB_SIZE'])

    preds = seed
    curr_pred = torch.topk(seed_pred[-1, :], k=1, dim=0)[1]
    curr_pred = chords_vocab_dict['idx2chord'][curr_pred.item()]
    preds.append(curr_pred)
    for t in range(num_chars):
        if len(intersection(preds[-4:], preds[-8:-4])) / len(preds[-4:]) >= 0.75:
            # print("random1")
            random_note = np.random.randint(0, len([x for x in chords_vocab_dict['chord2idx'].keys()]))
            curr_pred = [x for x in chords_vocab_dict['chord2idx'].keys()][random_note]  # or always same random_notes
        elif len(intersection(preds[-8:], preds[-16:-8])) / len(preds[-8:]) >= 0.75:
            # print("random2")
            random_note = np.random.randint(0, len([x for x in chords_vocab_dict['chord2idx'].keys()]))
            curr_pred = [x for x in chords_vocab_dict['chord2idx'].keys()][random_note]  # or always same random_notes
        else:
            curr_pred = curr_pred
        # print(curr_pred)
        curr_pred = MidiPreprocessor.prepare_sequence_chords([curr_pred], chords_vocab_dict['chord2idx'], onehot=False)
        curr_pred = curr_pred.reshape(1, len(inputs))
        curr_pred, state = model(curr_pred, state)

        curr_pred = curr_pred.reshape(-1, chords_vocab_dict['chord_VOCAB_SIZE'])
        curr_pred = torch.topk(curr_pred[-1, :], k=1, dim=0)[1]
        curr_pred = chords_vocab_dict['idx2chord'][curr_pred.item()]
        preds.append(curr_pred)
    return preds, curr_pred


def train_chords_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_manager = DataManager()
    data_loader = DataLoaderHandler(data_manager)

    formatted_data = data_manager.load_formatted_data()

    chords_vocab_dict = data_manager.load_chords_vocab_dicts()

    model_chord = LSTMT.LSTMT_chord(chords_vocab_dict['chord_VOCAB_SIZE']).to(device)
    optimizer_chord = optim.Adam(model_chord.parameters(), config["lr"])

    model_chord.to(device)
    NUM_EPOCHS = 100
    tr_loss = []
    avg_loss = None
    avg_weight = 0.1
    state = None
    timer_beg = timer()
    data_loader_x, data_loader_y = data_loader.get_chord_loaders(formatted_data['chords_duration_formatted_v2'])
    for epoch in range(NUM_EPOCHS):
        model_chord.train()
        for x, y in zip(data_loader_x, data_loader_y):
            # for x,y in zip(dataX,dataY):
            x = x.to(device)
            y = y.to(device)
            optimizer_chord.zero_grad()
            # Step 3. Run our forward pass.
            # Forward through model and carry the previous state forward in time (statefulness)
            y_, state = model_chord(x, state)
            # detach the previous state graph to not backprop gradients further than the BPTT span
            state = (state[0].detach(),  # detach c[t]
                     state[1].detach())  # detach h[t]

            y_ = y_.reshape(-1, chords_vocab_dict['chord_VOCAB_SIZE'])  # >> only when using batches
            loss = F.nll_loss(y_, y.view(-1))

            loss.backward()
            optimizer_chord.step()
            if avg_loss:
                avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
            else:
                avg_loss = loss.item()  # por que aqui no se suma y en la de test epoch se suma?
            tr_loss.append(loss.item())

        timer_end = timer()
        p = MODELS_PATH + "fake_chord_model_{}.pt".format(round(float(loss), 1))
        torch.save(model_chord.state_dict(), p)
        if (epoch + 1) % 5 == 0:
            # Generate a seed sentence to play around
            model_chord.to('cpu')
            print('-' * 30)
            print('-' * 30)
            model_chord.to(device)
            print('Finished epoch {} in {:.1f} s: loss: {:.6f}'.format(epoch + 1,
                                                                       timer_end - timer_beg,
                                                                       np.mean(tr_loss[-10:])))
        timer_beg = timer()


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_manager = DataManager()
    data_loader = DataLoaderHandler(data_manager)

    velocity_vocab_dict = data_manager.load_velocity_vocab_dicts()
    duration_vocab_dict = data_manager.load_notes_durations_vocab_dicts()
    formatted_data = data_manager.load_formatted_data()

    # model_chord.load_state_dict(torch.load("drive/MyDrive/UPC-Project/weights/chord_model_0.1.pt")) # anime

    model_two_embeddings = LSTMT.LSTMT_2embeddings(duration_vocab_dict['note_dur_VOCAB_SIZE'],
                                                   velocity_vocab_dict['vel_VOCAB_SIZE']).to(device)

    optimizer = optim.Adam(model_two_embeddings.parameters(), config["lr"])

    model_two_embeddings.to(device)
    NUM_EPOCHS = 50
    tr_loss = []
    avg_loss = None
    avg_weight = 0.1
    state = None
    timer_beg = timer()
    data_loader_x, data_loader_y = data_loader.get_note_velocity_loaders(
        formatted_data['notes_duration_velocity_formatted_v2'])
    for epoch in range(NUM_EPOCHS):
        model_two_embeddings.train()
        for x, y in zip(data_loader_x, data_loader_y):
            # for x,y in zip(dataX,dataY):
            x1 = x[:, 0].to(device)
            x2 = x[:, 1].to(device)
            y1 = y[:, 0].to(device)
            y2 = y[:, 1].to(device)

            optimizer.zero_grad()
            # Step 3. Run our forward pass.
            # Forward through model and carry the previous state forward in time (statefulness)
            y_1, y_2, state = model_two_embeddings(x1, x2, state)
            # detach the previous state graph to not backprop gradients further than the BPTT span
            state = (state[0].detach(),  # detach c[t]
                     state[1].detach())  # detach h[t]

            y_1 = y_1.reshape(-1, duration_vocab_dict['note_dur_VOCAB_SIZE'])  # >> only when using batches
            loss1 = F.nll_loss(y_1, y1.reshape(-1))

            y_2 = y_2.reshape(-1, velocity_vocab_dict['vel_VOCAB_SIZE'])  # >> only when using batches
            loss2 = F.nll_loss(y_2, y2.reshape(-1))

            loss = loss1 + loss2

            loss.backward()
            print(loss)

            optimizer.step()
            if avg_loss:
                avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
            else:
                avg_loss = loss.item()  # por que aqui no se suma y en la de test epoch se suma?
            tr_loss.append(loss.item())

        timer_end = timer()
        p = MODELS_PATH + "fake_test_two_embeddings_2_{}.pt".format(round(float(loss), 1))
        torch.save(model_two_embeddings.state_dict(), p)
        if (epoch + 1) % 5 == 0:
            # Generate a seed sentence to play around
            model_two_embeddings.to('cpu')
            print('-' * 30)
            print('-' * 30)
            model_two_embeddings.to(device)
            print('Finished epoch {} in {:.1f} s: loss: {:.6f}'.format(epoch + 1,
                                                                       timer_end - timer_beg,
                                                                       np.mean(tr_loss[-10:])))
        timer_beg = timer()

    # plt.plot(tr_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('NLLLoss')


def create_midi_chords(duration, model_name, song_name=None):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_manager = DataManager()
    chords_vocab_dict = data_manager.load_chords_vocab_dicts()
    model_chords = LSTMT.LSTMT_chord(chords_vocab_dict['chord_VOCAB_SIZE']).to(device)
    model_chords.load_state_dict(torch.load(MODELS_PATH + model_name))
    chord_output, dupa = gen_notes_chords(model_chords, duration)
    offset_chord = 0

    # TODO set the same tempo for chords and notes

    # set tempo
    temp = TEMPO
    output_notes = []
    chord_output_notes = []

    # Adding chords
    for pattern in chord_output:
        # pattern is a chord
        pattern_note = pattern.split("_")[0]
        pattern_duration = pattern.split("_")[1]

        if pattern_duration == "0.0":
            pattern_duration = "3.0"
        else:
            pattern_duration = pattern_duration

        if ('.' in pattern_note) or pattern_note.isdigit():
            notes_in_chord = pattern_note.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))

                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset_chord
            chord_output_notes.append(new_chord)
        # pattern is a note
        else:
            if pattern_note == 'Silence':
                new_note = note.Rest()
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))

            else:
                new_note = note.Note(pattern_note)
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))

                new_note.offset = offset_chord

                new_note.storedInstrument = instrument.Piano()
                chord_output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset_chord += Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator

    midi_stream_chord = stream.Part(chord_output_notes)
    midi_stream_chord.insert(0, instrument.Piano())
    midi_stream_chord.insert(tempo.MetronomeMark('lofi', temp))
    midi_stream_chord.insert(0, meter.TimeSignature('4/4'))

    if song_name is None:
        datetime = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        name = 'NV_{}_chords'.format(str(datetime))
    else:
        name = song_name + '_chords'
    midi_path = MIDI_PATH + name + '.mid'

    # Save Separated Chords and Notes tracks, as we will need different instruments SoundFonts for each.
    midi_stream_chord.write('midi', fp=midi_path)

    return name


def create_midi_notes(duration, model_name, song_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_manager = DataManager()


    #TODO fix typo
    valocity_vocab_dict = data_manager.load_velocity_vocab_dicts()
    duration_vocab_dict = data_manager.load_duration_vocab_dicts()
    notes_vocab_dict = data_manager.load_notes_vocab_dicts()

    model_three_embeddings = LSTMT.LSTMT_3embeddings(notes_vocab_dict['VOCAB_SIZE'], duration_vocab_dict['dur_VOCAB_SIZE'],
                                                     valocity_vocab_dict['vel_VOCAB_SIZE']).to(device)
    model_three_embeddings.load_state_dict(torch.load(MODELS_PATH + model_name))
    preds_notes = gen_notes_three_embeddings(model_three_embeddings, duration)

    offset = 0

    # set tempo
    temp = TEMPO
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in preds_notes:
        # pattern is a chord
        pattern_note = pattern.split("_")[0]
        pattern_duration = pattern.split("_")[1]

        if pattern_duration == "0.0":
            pattern_duration = "3.0"
        else:
            pattern_duration = pattern_duration
        pattern_velocity = pattern.split("_")[2]

        if ('.' in pattern_note) or pattern_note.isdigit():
            notes_in_chord = pattern_note.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))
                v = music21.volume.Volume(velocity=int(pattern_velocity))
                new_note.volume.velocity = v.velocity

                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            if pattern_note == 'Silence':
                new_note = note.Rest()
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))

            else:
                new_note = note.Note(pattern_note)
                new_note.duration = music21.duration.Duration(
                    (Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator))

                v = music21.volume.Volume(velocity=int(pattern_velocity))
                new_note.volume.velocity = v.velocity
                new_note.offset = offset

                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += Fraction(pattern_duration).numerator / Fraction(pattern_duration).denominator

    midi_stream = stream.Part(output_notes)
    midi_stream.insert(0, instrument.Piano())
    midi_stream.insert(tempo.MetronomeMark('lofi', temp))
    midi_stream.insert(0, meter.TimeSignature('4/4'))

    if song_name is None:
        datetime = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        name = 'NV_{}_chords'.format(str(datetime))
    else:
        name = song_name
    midi_path = MIDI_PATH + name + '.mid'

    midi_stream.write('midi', fp=midi_path)

    return name


def music_generator_starter():
    song_name = input('Provide the name of your song: ')
    include_chords = input('Your song should contain both chords and melody? yes/no: ')

    if include_chords == 'yes':
        include_chords = True
    else:
        print('Ok, then just a melody...')
        include_chords = False

    instruments_mapper = InstrumentsMapper()
    print("Available instruments:")
    print(instruments_mapper.get_all_instruments_names())

    melody_instrument = input('Select which instrument should play the melody in your song: ')
    if include_chords:
        chords_instrument = input('Select which instrument should play the chords in your song: ')
    else:
        chords_instrument = None

    effects_mapper = EffectsMapper()
    print("Available effects:")
    print(effects_mapper.get_all_effect_names())

    effects = input("Select effects separated by comma: ")
    print('Alright, give me a sec! :D')

    result_set = {'song_name': song_name,
                  'include_chords': include_chords,
                  'melody_instrument': melody_instrument,
                  'chords_instrument': chords_instrument,
                  'effects': effects}

    return result_set


if __name__ == '__main__':

    # possible modes: 'train_main', 'train_chords', 'generate_notes', 'generate_chords'
    mode = 'generate_music'

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print(mode)

    if mode == 'train_notes':
        print('Notes model started training... Press Ctrl + C to stop')
        train_model()

    elif mode == 'train_chords':
        print('Chords model started training... Press Ctrl + C to stop')
        train_chords_model()
    elif mode == 'test':
        notes_midi_name = create_midi_notes(GEN_SIZE, "big_test_THREE_embeddings_0.9.pt",
                                            'dupa')
    elif mode == 'generate_music':
        print('Welcome to Lofi music generator!')
        user_choices = music_generator_starter()

        notes_midi_name = create_midi_notes(GEN_SIZE, "big_test_THREE_embeddings_0.9.pt",
                                            user_choices['song_name'])
        if user_choices['include_chords']:
            chords_midi_name = create_midi_chords(GEN_SIZE, "chord_model_0.1.pt",
                                                  user_choices['song_name'])

        midi2Wav = MidiToWav(PATH)
        midi2Wav.set_name(user_choices['song_name'])
        midi2Wav.midi_to_wav(user_choices['melody_instrument'])

        if user_choices['include_chords']:
            midi2Wav.add_chords(user_choices['chords_instrument'], chords_midi_name)

        midi2Wav.post_effects(user_choices['effects'].split(','))

        print('All done! Check out the app/data/final_results dir!')
