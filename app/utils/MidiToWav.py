import os

from midi2audio import FluidSynth

from pydub import AudioSegment
import fluidsynth

from app.utils.mappers.EffectsMapper import EffectsMapper
from app.utils.mappers.InstrumentsMapper import InstrumentsMapper


class MidiToWav:

    def __init__(self, root_path):
        self.instrument_chords = 'Super_Mario_Advance_4.sf2'  # default value
        self.instrument_notes = 'Super_Nintendo_Unofficial_update.sf2'  # default value
        self.name = None
        self.root_path = root_path
        self.midi_path = root_path + '/midi/'
        self.wav_path = root_path + '/wav/'
        self.final_result_path = root_path + '/final_results/'
        self.sf2_path = root_path + '/sf2/'
        self.effects_path = root_path + '/effects/'
        self.effects_mapper = EffectsMapper()
        self.instruments_mapper = InstrumentsMapper()

    def convert_midi_to_wav(self):
        FluidSynth(self.path + '/rhodes.sf2').midi_to_audio(
            self.path + 'Model_chord_and_notes_velocity_chord_velocity_model_test_notes.mid',
            self.path + 'Model_chord_and_notes_velocity_chord_velocity_model_test_notes.wav')

        sound1 = AudioSegment.from_file(
            self.path + 'Model_chord_and_notes_velocity_chord_velocity_model_test_notes.wav')
        sound2 = AudioSegment.from_file(
            self.path + 'Model_chord_and_notes_velocity_chord_velocity_model_test_chords.wav')
        # sound3 = AudioSegment.from_file(midi_path[:-4]+'_chord_converted.wav')

        # modify track sound
        sound1 = sound1 - 5  # +10
        sound2 = sound2 - 5  # -5 0

        combined = sound1.overlay(sound2)

        combined.export(self.path + 'combined1.wav', format='wav'),

        # comby two tracks
        sound3 = AudioSegment.from_file(self.path + 'rain-06.mp3'),
        sound4 = AudioSegment.from_file(self.path + 'slow_jam_beat.wav'),

        # modify track sound
        sound3 = sound3 - 10
        sound4 = sound4
        # sound3 = sound3 -25
        combined2 = sound3.overlay(sound4)

        combined2.export(self.path + 'combined2.wav', format='wav')

        combined2

        sound5 = AudioSegment.from_file(self.path + 'combined1.wav')
        sound6 = AudioSegment.from_file(self.path + 'combined2.wav')

        sound5 = sound5
        sound6 = sound6 - 10

        combined3 = sound5.overlay(sound6)
        combined3.export(self.path + 'combined_final.wav', format='wav')

    def midi_to_wav(self, instrument):
        FluidSynth(self.sf2_path + self.instruments_mapper.get_instrument_file_name(instrument)).midi_to_audio(
            self.midi_path + self.name + '.mid',
            self.wav_path + self.name + '.wav')

    def add_chords(self, instrument, chords_midi_name):
        FluidSynth(self.sf2_path + self.instruments_mapper.get_instrument_file_name(instrument)).midi_to_audio(
            self.midi_path + chords_midi_name + '.mid',
            self.wav_path + chords_midi_name + '.wav')

        notes = AudioSegment.from_file(self.wav_path + self.name + '.wav')
        chords = AudioSegment.from_file(self.wav_path + chords_midi_name + '.wav')

        merged = notes.overlay(chords)

        merged.export(self.wav_path + self.name + '.wav', format='wav')

    def post_effects(self, effects_list):
        sound = AudioSegment.from_file(self.wav_path + self.name + '.wav')

        for effect in effects_list:
            sound = sound.overlay(AudioSegment.from_file(self.effects_path +
                                                         self.effects_mapper.get_effect_file_name(effect)))

        sound.export(self.final_result_path + self.name + '.wav', format='wav')

    def set_name(self, name):
        self.name = name


if __name__ == '__main__':
    file_name = 'Model_chord_and_notes_velocity_3_big_embs_1'
    midiWav = MidiToWav(os.getcwd() + '/../data')

    for i in InstrumentsMapper().get_all_instruments_names():
        print(i)

    instrument = input("choose instrument: ")

    midiWav.set_name(file_name)
    midiWav.midi_to_wav(instrument)

    for e in EffectsMapper().get_all_effect_names():
        print(e)

    effects = input("choose effects separated by comma: ")

    midiWav.post_effects(effects.split(','))
