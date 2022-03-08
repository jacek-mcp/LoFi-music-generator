import numpy as np
import torch


class MidiPreprocessor:

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

    # for two embeddings model
    @staticmethod
    def prepare_sequence(seq, char2idx, dur2idx, vel2idx, onehot=False):
        # 3 embeddings version
        # convert sequence of words to indices
        seq = [x.split("_") for x in seq]

        # idxs_notes_dur = [char2idx_note_dur["_".join([c[0], str(c[1])])] for c in seq]
        # idxs_notes_dur = torch.tensor(idxs_notes_dur, dtype=torch.long)

        idxs_notes = [char2idx[c[0]] for c in seq]
        idxs_notes = torch.tensor(idxs_notes, dtype=torch.long)

        idxs_dur = [dur2idx[c[1]] for c in seq]
        idxs_dur = torch.tensor(idxs_dur, dtype=torch.long)

        idxs_vel = [vel2idx[c[2]] for c in seq]
        idxs_vel = torch.tensor(idxs_vel, dtype=torch.long)

        return [idxs_notes, idxs_dur, idxs_vel]

    @staticmethod
    def prepare_sequence_chords(seq, chord2idx, onehot=False):
        # convert sequence of words to indices
        seq = [x.split("_") for x in seq]

        idxs_chords_dur = [chord2idx["_".join([c[0], str(c[1])])] for c in seq]
        idxs_chords_dur = torch.tensor(idxs_chords_dur, dtype=torch.long)

        return idxs_chords_dur

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