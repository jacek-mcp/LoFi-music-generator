import pickle
import os


class DataManager:

    def __init__(self):

        self.path = os.getcwd() + '/app/data/data/'

    # def format_and_save_final_data(self, notes_list, duration_list_formatted, velocity_list_formatted, chords_notes,
    #                                chords_duaration_formatted):
    #     notes_duration_note_velocity_formatted_v2 = ["_".join([a, str(b), a, str(c)]) for a, b, c in
    #                                                  zip(notes_list[0], duration_list_formatted,
    #                                                      velocity_list_formatted)]
    #     notes_duration_velocity_formatted_v2 = ["_".join([a, str(b), str(c)]) for a, b, c in
    #                                             zip(notes_list[0], duration_list_formatted, velocity_list_formatted)]
    #     notes_velocity_formatted_v2 = ["_".join([a, str(c)]) for a, b, c in
    #                                    zip(notes_list[0], duration_list_formatted, velocity_list_formatted)]
    #     notes_duration_formatted_v2 = ["_".join([a, str(b)]) for a, b in zip(notes_list[0], duration_list_formatted)]
    #     velocity_list_formatted_v2 = ["_".join([str(a)]) for a in velocity_list_formatted]
    #     chords_duration_formatted_v2 = ["_".join([a, str(b)]) for a, b in
    #                                     zip(chords_notes[0], chords_duaration_formatted)]
    #     notes_formatted_v2 = [a for a in notes_list[0]]
    #     duration_formatted_v2 = [a for a in duration_list_formated]
    #
    #     with open(self.path + 'big_anime_final_notes_dur_note_vel', 'wb') as filepath:
    #         pickle.dump(notes_duration_note_velocity_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_notes_dur_vel', 'wb') as filepath:
    #         pickle.dump(notes_duration_velocity_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_notes_dur', 'wb') as filepath:
    #         pickle.dump(notes_duration_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_notes_vel', 'wb') as filepath:
    #         pickle.dump(notes_velocity_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_vel', 'wb') as filepath:
    #         pickle.dump(velocity_list_formatted_v2, filepath)
    #
    #     with open(self.path + 'small_anime_final_chord_dur', 'wb') as filepath:
    #         pickle.dump(chords_duration_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_notes_v2', 'wb') as filepath:
    #         pickle.dump(notes_formatted_v2, filepath)
    #
    #     with open(self.path + 'big_anime_final_duration_v2', 'wb') as filepath:
    #         pickle.dump(duration_formatted_v2, filepath)
    #
    #     result_set = {'notes_duration_note_velocity_formatted_v2': notes_duration_note_velocity_formatted_v2,
    #                   'notes_duration_velocity_formatted_v2': notes_duration_velocity_formatted_v2,
    #                   'notes_velocity_formatted_v2': notes_velocity_formatted_v2,
    #                   'notes_duration_formatted_v2': notes_duration_formatted_v2,
    #                   'velocity_list_formatted_v2': velocity_list_formatted_v2,
    #                   'chords_duration_formatted_v2': chords_duration_formatted_v2,
    #                   'notes_formatted_v2': notes_formatted_v2,
    #                   'duration_formatted_v2': duration_formatted_v2}
    #
    #     return result_set

    def load_formatted_data(self):
        with open(self.path + 'big_anime_final_notes_dur_note_vel_v2',
                  'rb') as filepath:  # big_anime_final_notes_dur_note_vel_v2
            notes_duration_note_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_notes_dur_vel_v2', 'rb') as filepath:  # big_anime_final_notes_dur_vel_v2
            notes_duration_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_notes_dur_v2', 'rb') as filepath:  # big_anime_final_notes_dur_v2
            notes_duration_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_notes_vel_v2', 'rb') as filepath:  # big_anime_final_notes_vel_v2
            notes_velocity_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_vel_v2', 'rb') as filepath:  # big_anime_final_vel_v2
            velocity_list_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_chord_dur_v2', 'rb') as filepath:  # big_anime_final_chord_dur_v2

            chords_duration_formatted_v2 = pickle.load(filepath)
            
  
        with open(self.path + 'big_anime_final_notes_v2', 'rb') as filepath: 
            notes_formatted_v2 = pickle.load(filepath)
            
        with open(self.path + 'big_anime_final_duration_v2', 'rb') as filepath: 
            duration_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_notes_v2', 'rb') as filepath:
            notes_formatted_v2 = pickle.load(filepath)

        with open(self.path + 'big_anime_final_duration_v2', 'rb') as filepath:
            duration_formatted_v2 = pickle.load(filepath)

        result_set = {'notes_duration_note_velocity_formatted_v2': notes_duration_note_velocity_formatted_v2,
                      'notes_duration_velocity_formatted_v2': notes_duration_velocity_formatted_v2,
                      'notes_velocity_formatted_v2': notes_velocity_formatted_v2,
                      'notes_duration_formatted_v2': notes_duration_formatted_v2,
                      'velocity_list_formatted_v2': velocity_list_formatted_v2,
                      'chords_duration_formatted_v2': chords_duration_formatted_v2,
                      'notes_formatted_v2': notes_formatted_v2,
                      'duration_formatted_v2': duration_formatted_v2}

        return result_set

    def load_notes_old(self):
        with open(self.path + 'big_notes_big', 'rb') as filepath:
            notes = pickle.load(filepath)
        with open(self.path + 'small_anime_notes', 'rb') as filepath:
            notes_list = pickle.load(filepath)

        with open(self.path + 'big_anime_velocity', 'rb') as filepath:
            velocity_list = pickle.load(filepath)

        with open(self.path + 'big_anime_notes_duration', 'rb') as filepath:
            notes_duration_list = pickle.load(filepath)

        with open(self.path + 'big_anime_notes_velocity', 'rb') as filepath:
            notes_velocity_list = pickle.load(filepath)

        with open(self.path + 'big_anime_durations', 'rb') as filepath:
            duration_list = pickle.load(filepath)

        with open(self.path + 'small_anime_chords_note', 'rb') as filepath:
            chords_notes = pickle.load(filepath)

        with open(self.path + 'small_anime_chords_duration', 'rb') as filepath:
            chords_duaration = pickle.load(filepath)

        result_set = {'notes': notes,
                      'notes_list': notes_list,
                      'notes_duration_list': notes_duration_list,
                      'notes_velocity_list': notes_velocity_list,
                      'duration_list': duration_list,
                      'chords_notes': chords_notes,
                      'chords_duaration': chords_duaration}

        return result_set

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
        with open(self.path + 'big_vel2idx', 'rb') as filepath:
            vel2idx = pickle.load(filepath)

        with open(self.path + 'big_idx2vel', 'rb') as filepath:
            idx2vel = pickle.load(filepath)

        with open(self.path + 'big_vel_VOCAB_SIZE', 'rb') as filepath:
            vel_VOCAB_SIZE = pickle.load(filepath)

        result_set = {'vel2idx': vel2idx,
                      'idx2vel': idx2vel,
                      'vel_VOCAB_SIZE': vel_VOCAB_SIZE}

        return result_set

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

        result_set = {'chord2idx': chord2idx,
                      'idx2chord': idx2chord,
                      'chord_VOCAB_SIZE': chord_VOCAB_SIZE}

        return result_set

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

        result_set = {'chord2idx': chord2idx,
                      'idx2chord': idx2chord,
                      'chord_VOCAB_SIZE': chord_VOCAB_SIZE}

        return result_set

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

        result_set = {'char2idx_note_dur': char2idx_note_dur,
                      'idx2char_note_dur': idx2char_note_dur,
                      'note_dur_VOCAB_SIZE': note_dur_VOCAB_SIZE}

        return result_set

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

        result_set = {'char2idx_note_dur': char2idx_note_dur,
                      'idx2char_note_dur': idx2char_note_dur,
                      'note_dur_VOCAB_SIZE': note_dur_VOCAB_SIZE}

        return result_set
    
  
    
    def create_and_save_notes_vocab_dicts(self, notes_formatted_v2):
        # merge the training data into one big text line

        training_data = [notes_formatted_v2]
        char2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in char2idx:
                    char2idx[c] = len(char2idx)
        idx2char = dict((v, k) for k, v in char2idx.items())
        VOCAB_SIZE = len(char2idx)
        print('Number of found vocabulary tokens: ', VOCAB_SIZE)

        with open(self.path + 'big_char2idx', 'wb') as filepath:
            pickle.dump(char2idx, filepath)

        with open(self.path + 'big_idx2char', 'wb') as filepath:
            pickle.dump(idx2char, filepath)

        with open(self.path + 'big_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(VOCAB_SIZE, filepath)

        result_set = {'char2idx': char2idx,
                      'idx2char': idx2char,
                      'VOCAB_SIZE': VOCAB_SIZE}

        return result_set

    def load_notes_vocab_dicts(self):
        with open(self.path + 'big_char2idx', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            char2idx = pickle.load(filepath)

        with open(self.path + 'big_idx2char', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2char = pickle.load(filepath)

        with open(self.path + 'big_VOCAB_SIZE', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            VOCAB_SIZE = pickle.load(filepath)

        result_set = {'char2idx': char2idx_note_dur,
                      'idx2char': idx2char_note_dur,
                      'VOCAB_SIZE': note_dur_VOCAB_SIZE}

        return result_set
  
    
    
    def create_and_save_duration_vocab_dicts(self, duration_formatted_v2):
        # merge the training data into one big text line

        training_data = [duration_formatted_v2]
        dur2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in dur2idx:
                    dur2idx[c] = len(dur2idx)
        idx2dur = dict((v, k) for k, v in dur2idx.items())
        dur_VOCAB_SIZE = len(dur2idx)
        print('Number of found vocabulary tokens: ', dur_VOCAB_SIZE)

        with open(self.path + 'big_dur2idx', 'wb') as filepath:
            pickle.dump(dur2idx, filepath)

        with open(self.path + 'big_idx2dur', 'wb') as filepath:
            pickle.dump(idx2dur, filepath)

        with open(self.path + 'big_dur_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(dur_VOCAB_SIZE, filepath)

        result_set = {'dur2idx': dur2idx,
                      'idx2dur': idx2dur,
                      'dur_VOCAB_SIZE': dur_VOCAB_SIZE}

        return result_set

    def load_duration_vocab_dicts(self):
        with open(self.path + 'big_dur2idx', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            dur2idx = pickle.load(filepath)

        with open(self.path + 'big_idx2dur', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2dur = pickle.load(filepath)

        with open(self.path + 'big_dur_VOCAB_SIZE', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            dur_VOCAB_SIZE = pickle.load(filepath)

        result_set = {'dur2idx': dur2idx,
                      'idx2dur': idx2dur,
                      'dur_VOCAB_SIZE': dur_VOCAB_SIZE}

        return result_set
  
    
    

    def create_and_save_notes_vocab_dicts(self, notes_formatted_v2):
        # merge the training data into one big text line

        training_data = [notes_formatted_v2]
        char2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in char2idx:
                    char2idx[c] = len(char2idx)
        idx2char = dict((v, k) for k, v in char2idx.items())
        VOCAB_SIZE = len(char2idx)
        print('Number of found vocabulary tokens: ', VOCAB_SIZE)

        with open(self.path + 'big_char2idx', 'wb') as filepath:
            pickle.dump(char2idx, filepath)

        with open(self.path + 'big_idx2char', 'wb') as filepath:
            pickle.dump(idx2char, filepath)

        with open(self.path + 'big_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(VOCAB_SIZE, filepath)

        result_set = {'char2idx': char2idx,
                      'idx2char': idx2char,
                      'VOCAB_SIZE': VOCAB_SIZE}

        return result_set

    def load_notes_vocab_dicts(self):
        with open(self.path + 'big_char2idx_v2', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            char2idx = pickle.load(filepath)

        with open(self.path + 'big_idx2char_v2', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2char = pickle.load(filepath)

        with open(self.path + 'big_VOCAB_SIZE_v2', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            VOCAB_SIZE = pickle.load(filepath)

        result_set = {'char2idx': char2idx,
                      'idx2char': idx2char,
                      'VOCAB_SIZE': VOCAB_SIZE}

        return result_set

    def create_and_save_duration_vocab_dicts(self, duration_formatted_v2):
        # merge the training data into one big text line

        training_data = [duration_formatted_v2]
        dur2idx = {}
        for sent in training_data:
            for c in sent:
                if c not in dur2idx:
                    dur2idx[c] = len(dur2idx)
        idx2dur = dict((v, k) for k, v in dur2idx.items())
        dur_VOCAB_SIZE = len(dur2idx)
        print('Number of found vocabulary tokens: ', dur_VOCAB_SIZE)

        with open(self.path + 'big_dur2idx', 'wb') as filepath:
            pickle.dump(dur2idx, filepath)

        with open(self.path + 'big_idx2dur', 'wb') as filepath:
            pickle.dump(idx2dur, filepath)

        with open(self.path + 'big_dur_VOCAB_SIZE', 'wb') as filepath:
            pickle.dump(dur_VOCAB_SIZE, filepath)

        result_set = {'dur2idx': dur2idx,
                      'idx2dur': idx2dur,
                      'dur_VOCAB_SIZE': dur_VOCAB_SIZE}

        return result_set

    def load_duration_vocab_dicts(self):
        with open(self.path + 'big_dur2idx', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            dur2idx = pickle.load(filepath)

        with open(self.path + 'big_idx2dur', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            idx2dur = pickle.load(filepath)

        with open(self.path + 'big_dur_VOCAB_SIZE', 'rb') as filepath:
            # with open('data/lofi_notes_velocity', 'rb') as filepath:
            dur_VOCAB_SIZE = pickle.load(filepath)

        result_set = {'dur2idx': dur2idx,
                      'idx2dur': idx2dur,
                      'dur_VOCAB_SIZE': dur_VOCAB_SIZE}

        return result_set

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

        result_set = {'char2idx_note_dur_vel': char2idx_note_dur_vel,
                      'idx2char_note_dur_vel': idx2char_note_dur_vel,
                      'note_dur_vel_VOCAB_SIZE': note_dur_vel_VOCAB_SIZE}

        return result_set

    def load_notes_duration_velocity_vocab_dicts(self):
        with open(self.path + 'big_char2idx_note_dur_vel', 'rb') as filepath:
            char2idx_note_dur_vel = pickle.load(filepath)

        with open(self.path + 'big_idx2char_note_dur_vel', 'rb') as filepath:
            idx2char_note_dur_vel = pickle.load(filepath)

        with open(self.path + 'big_note_dur_vel_VOCAB_SIZE', 'rb') as filepath:
            note_dur_vel_VOCAB_SIZE = pickle.load(filepath)

        result_set = {'char2idx_note_dur_vel': char2idx_note_dur_vel,
                      'idx2char_note_dur_vel': idx2char_note_dur_vel,
                      'note_dur_vel_VOCAB_SIZE': note_dur_vel_VOCAB_SIZE}

        return result_set