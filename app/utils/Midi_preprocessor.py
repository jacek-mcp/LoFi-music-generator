import pickle

import numpy as np
import torch
import torch.nn.functional as F


class MidiPreprocessor:

    def __init__(self, path):
        self.path = path

    @staticmethod
    def normalize_durations(duration_list):
        # Note Duration Normalization
        lofi_bins = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        centers = (lofi_bins[1:] + lofi_bins[:-1]) / 2
        duration_list_formatted = [str(lofi_bins[np.digitize(round(x, 2), centers)]) for x in duration_list[0]]
        return duration_list_formatted

    @staticmethod
    def normalize_durations(velocity_list):
        # Velocity Normalization
        bins_vel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 200])
        centers = (bins_vel[1:] + bins_vel[:-1]) / 2
        velocity_list_formatted = [str(bins_vel[np.digitize(round(int(x), 2), centers)]) for x in velocity_list[0]]
        return velocity_list_formatted

    def format_and_save_final_data(self, notes_list, duration_list_formatted, velocity_list_formatted, chords_notes,
                                   chords_duaration_formatted):
        notes_duration_note_velocity_formatted_v2 = ["_".join([a, str(b), a, str(c)]) for a, b, c in
                                                     zip(notes_list[0], duration_list_formatted,
                                                         velocity_list_formatted)]
        notes_duration_velocity_formatted_v2 = ["_".join([a, str(b), str(c)]) for a, b, c in
                                                zip(notes_list[0], duration_list_formatted, velocity_list_formatted)]
        notes_velocity_formatted_v2 = ["_".join([a, str(c)]) for a, b, c in
                                       zip(notes_list[0], duration_list_formatted, velocity_list_formatted)]
        notes_duration_formatted_v2 = ["_".join([a, str(b)]) for a, b in zip(notes_list[0], duration_list_formatted)]
        velocity_list_formatted_v2 = ["_".join([str(a)]) for a in velocity_list_formatted]
        chords_duration_formatted_v2 = ["_".join([a, str(b)]) for a, b in
                                        zip(chords_notes[0], chords_duaration_formatted)]

        with open(self.path + 'small_anime_final_notes_dur_note_vel', 'wb') as filepath:
            pickle.dump(notes_duration_note_velocity_formatted_v2, filepath)

        with open(self.path + 'small_anime_final_notes_dur_vel', 'wb') as filepath:
            pickle.dump(notes_duration_velocity_formatted_v2, filepath)

        with open(self.path + 'small_anime_final_notes_dur', 'wb') as filepath:
            pickle.dump(notes_duration_formatted_v2, filepath)

        with open(self.path + 'small_anime_final_notes_vel', 'wb') as filepath:
            pickle.dump(notes_velocity_formatted_v2, filepath)

        with open(self.path + 'small_anime_final_vel', 'wb') as filepath:
            pickle.dump(velocity_list_formatted_v2, filepath)

        with open(self.path + 'small_anime_final_chord_dur', 'wb') as filepath:
            pickle.dump(chords_duration_formatted_v2, filepath)

        return notes_duration_note_velocity_formatted_v2, notes_duration_velocity_formatted_v2, \
               notes_velocity_formatted_v2, notes_duration_formatted_v2, velocity_list_formatted_v2, chords_duration_formatted_v2

    def load_formatted_data(self):
        with open(self.path + 'small_anime_final_notes_dur_note_vel', 'rb') as filepath:
            notes_duration_note_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'small_anime_final_notes_dur_vel', 'rb') as filepath:
            notes_duration_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'small_anime_final_notes_dur', 'rb') as filepath:  # augmented_
            notes_duration_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'small_anime_final_notes_vel', 'rb') as filepath:
            notes_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'small_anime_final_vel', 'rb') as filepath:
            velocity_list_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'small_anime_final_chord_dur', 'rb') as filepath:
            chords_duration_formatted_v2 = pickle.load(filepath)

            return notes_duration_note_velocity_formatted_v2, notes_duration_velocity_formatted_v2, \
                   notes_duration_formatted_v2, notes_velocity_formatted_v2, velocity_list_formatted_v2, chords_duration_formatted_v2

    def load_notes_old(self):
        with open(self.path + 'small_notes_big', 'rb') as filepath:
            notes = pickle.load(filepath)
        with open(self.path + 'small_anime_notes', 'rb') as filepath:
            notes_list = pickle.load(filepath)

        with open(self.path + 'small_anime_velocity', 'rb') as filepath:
            velocity_list = pickle.load(filepath)

        with open(self.path + 'small_anime_notes_duration', 'rb') as filepath:
            notes_duration_list = pickle.load(filepath)

        with open(self.path + 'small_anime_notes_velocity', 'rb') as filepath:
            notes_velocity_list = pickle.load(filepath)

        with open(self.path + 'small_anime_durations', 'rb') as filepath:
            duration_list = pickle.load(filepath)

        with open(self.path + 'small_anime_chords_note', 'rb') as filepath:
            chords_notes = pickle.load(filepath)

        with open(self.path + 'small_anime_chords_duration', 'rb') as filepath:
            chords_duaration = pickle.load(filepath)

        return notes, notes_list, velocity_list, notes_duration_list, notes_velocity_list, duration_list, chords_notes, chords_duaration

    # for two embeddings model
    @staticmethod
    def prepare_sequence(seq, vel2idx, char2idx_note_dur, onehot=False):
        # convert sequence of words to indices
        seq = [x.split("_") for x in seq]

        idxs_notes_dur = [char2idx_note_dur["_".join([c[0], str(c[1])])] for c in seq]
        idxs_notes_dur = torch.tensor(idxs_notes_dur, dtype=torch.long)

        idxs_vel = [vel2idx[c[2]] for c in seq]
        idxs_vel = torch.tensor(idxs_vel, dtype=torch.long)

        return [idxs_notes_dur, idxs_vel]

    @staticmethod
    def prepare_sequence_chords(seq, chord2idx, onehot=False):
        # convert sequence of words to indices
        seq = [x.split("_") for x in seq]

        idxs_chords_dur = [chord2idx["_".join([c[0], str(c[1])])] for c in seq]
        idxs_chords_dur = torch.tensor(idxs_chords_dur, dtype=torch.long)

        return idxs_chords_dur

    def create_and_save_velocity_vocab_dicts(self, velocity_list_formated_v2):
        # merge the training data into one big text line
        training_data = [
            velocity_list_formated_v2]  # notes_duration_formated_v2 /  notes_duration_list  notes_duration_list
        # training_data = '_'.join(training_data[0])

        # Assign a unique ID to each different character found in the training set
        vel2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in vel2idx:
                    vel2idx[c] = len(vel2idx)
        idx2vel = dict((v, k) for k, v in vel2idx.items())
        vel_VOCAB_SIZE = len(vel2idx)
        print('Number of found vocabulary tokens: ', vel_VOCAB_SIZE)

        with open(self.path + 'small_vel2idx', 'wb') as filepath:
            pickle.dump(vel2idx, filepath)

        with open(self.path + 'small_idx2vel', 'wb') as filepath:
            pickle.dump(idx2vel, filepath)

        with open(self.path + 'small_vel_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(vel_VOCAB_SIZE, filepath)

        return vel2idx, idx2vel, vel_VOCAB_SIZE

    def load_velocity_vocab_dicts(self):
        with open(self.path + 'small_vel2idx', 'rb') as filepath:
            vel2idx = pickle.load(filepath)

        with open(self.path + 'small_idx2vel', 'rb') as filepath:
            idx2vel = pickle.load(filepath)

        with open(self.path + 'small_vel_VOCAB_SIZE', 'rb') as filepath:
            vel_VOCAB_SIZE = pickle.load(filepath)

        return vel2idx, idx2vel, vel_VOCAB_SIZE

    def create_and_save_chords_vocab_dicts(self, chords_duration_formated_v2):
        training_data = [chords_duration_formated_v2]
        chord2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in chord2idx:
                    chord2idx[c] = len(chord2idx)
        idx2chord = dict((v, k) for k, v in chord2idx.items())
        chord_VOCAB_SIZE = len(chord2idx)
        print('Number of found vocabulary tokens: ', chord_VOCAB_SIZE)

        with open(self.path + 'small_chord2idx', 'wb') as filepath:
            pickle.dump(chord2idx, filepath)

        with open(self.path + 'small_idx2chord', 'wb') as filepath:
            pickle.dump(idx2chord, filepath)

        with open(self.path + 'small_chord_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(chord_VOCAB_SIZE, filepath)

        return chord2idx, idx2chord, chord_VOCAB_SIZE

    def load_chords_vocab_dicts(self):
        with open(self.path + 'small_chord2idx', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            chord2idx = pickle.load(filepath)

        with open(self.path + 'small_idx2chord', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2chord = pickle.load(filepath)

        with open(self.path + 'small_chord_VOCAB_SIZE', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            chord_VOCAB_SIZE = pickle.load(filepath)

            return chord2idx, idx2chord, chord_VOCAB_SIZE

    def create_and_save_notes_durations_vocab_dicts(self, notes_duration_formated_v2):
        # merge the training data into one big text line

        training_data = [notes_duration_formated_v2]
        char2idx_note_dur = {}
        for sent in training_data:
            for c in sent:
                if c not in char2idx_note_dur:
                    char2idx_note_dur[c] = len(char2idx_note_dur)
        idx2char_note_dur = dict((v, k) for k, v in char2idx_note_dur.items())
        note_dur_VOCAB_SIZE = len(char2idx_note_dur)
        print('Number of found vocabulary tokens: ', note_dur_VOCAB_SIZE)

        with open(self.path + 'small_char2idx_note_dur', 'wb') as filepath:
            pickle.dump(char2idx_note_dur, filepath)

        with open(self.path + 'small_idx2char_note_dur', 'wb') as filepath:
            pickle.dump(idx2char_note_dur, filepath)

        with open(self.path + 'small_note_dur_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(note_dur_VOCAB_SIZE, filepath)

        return char2idx_note_dur, idx2char_note_dur, note_dur_VOCAB_SIZE

    def load_notes_durations_vocab_dicts(self):
        with open(self.path + 'small_char2idx_note_dur', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            char2idx_note_dur = pickle.load(filepath)

        with open(self.path + 'small_idx2char_note_dur', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2char_note_dur = pickle.load(filepath)

        with open(self.path + 'small_note_dur_VOCAB_SIZE', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            note_dur_VOCAB_SIZE = pickle.load(filepath)

        return char2idx_note_dur, idx2char_note_dur, note_dur_VOCAB_SIZE


    #TODO make loader of those files
    def create_and_save_notes_duration_velocity_vocab_dicts(self, notes_duration_velocity_formated_v2):
        # merge the training data into one big text line
        training_data = [
            notes_duration_velocity_formated_v2]  # notes_duration_formated_v2 /  notes_duration_list  notes_duration_list

        char2idx_note_dur_vel = {}
        for sent in training_data:
            for c in sent:
                if c not in char2idx_note_dur_vel:
                    char2idx_note_dur_vel[c] = len(char2idx_note_dur_vel)
        idx2char_note_dur_vel = dict((v, k) for k, v in char2idx_note_dur_vel.items())
        note_dur_vel_VOCAB_SIZE = len(char2idx_note_dur_vel)
        print('Number of found vocabulary tokens: ', note_dur_vel_VOCAB_SIZE)

        with open(self.path + 'small_char2idx_note_dur_vel', 'wb') as filepath:
            pickle.dump(char2idx_note_dur_vel, filepath)

        with open(self.path + 'small_idx2char_note_dur_vel', 'wb') as filepath:
            pickle.dump(idx2char_note_dur_vel, filepath)

        with open(self.path + 'small_note_dur_vel_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(note_dur_vel_VOCAB_SIZE, filepath)

        return char2idx_note_dur_vel, idx2char_note_dur_vel, note_dur_vel_VOCAB_SIZE


    def load_notes_duration_velocity_vocab_dicts(self):
        with open(self.path + 'small_char2idx_note_dur_vel', 'rb') as filepath:
            char2idx_note_dur_vel = pickle.load(filepath)

        with open(self.path + 'small_idx2char_note_dur_vel', 'rb') as filepath:
            idx2char_note_dur_vel = pickle.load(filepath)

        with open(self.path + 'small_note_dur_vel_VOCAB_SIZE', 'rb') as filepath:
            note_dur_vel_VOCAB_SIZE = pickle.load(filepath)

        return char2idx_note_dur_vel, idx2char_note_dur_vel, note_dur_vel_VOCAB_SIZE

    @staticmethod
    def normalize_durations(duration_list):
        # Note Duration Normalization
        lofi_bins = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        centers = (lofi_bins[1:] + lofi_bins[:-1]) / 2
        duration_list_formated = [str(lofi_bins[np.digitize(round(x, 2), centers)]) for x in duration_list[0]]
        return duration_list_formated

    @staticmethod
    def normalize_durations_chords(chord_duration_list):
        # Chord Duration Normalization
        lofi_bins = np.array([2.0, 3.0, 4.0])
        centers = (lofi_bins[1:] + lofi_bins[:-1]) / 2
        duration_list_formated = [str(lofi_bins[np.digitize(round(float(x), 2), centers)]) for x in
                                  chord_duration_list[0]]
        return duration_list_formated

    @staticmethod
    def normalize_velocity(velocity_list):
        # Velocity Normalization
        bins_vel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 200])
        centers = (bins_vel[1:] + bins_vel[:-1]) / 2
        velocity_list_formated = [str(bins_vel[np.digitize(round(int(x), 2), centers)]) for x in velocity_list[0]]
        return velocity_list_formated
