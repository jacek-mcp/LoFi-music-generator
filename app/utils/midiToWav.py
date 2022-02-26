import os

from midi2audio import FluidSynth

from pydub import AudioSegment


class midiToWav:

    def __init__(self):
        self.path = os.getcwd() + '/data/midiToWav'

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

if __name__ == '__main__':

    midiWav = midiToWav()

    midiWav.convert_midi_to_wav()