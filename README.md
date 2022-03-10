# LoFi-music-generator
An open-sourced Lofi hip-hop music generation project for UPC. 

Table of Contents
=================

  * [About The Project](#introduction-and-motivation)
  * [DATASET](#dataset)
  * [ARCHITECTURES](#architecture-and-results)
  * [LSTM](#classifier-neural-network)
  * [MODEL IMPROVEMENTS](#model-improvements)
  * [Data augmentation](#data-augmentation---trial-i)
  * [End to end generator](#end-to-end-generator)
  * [Prerequisites](#prerequisites)
  * [Build & Run](#build--run)
  * [How to add your post effects & instruments](#how-to-add-your-post-effects--instruments)
	 

## About The Project

The goal of this project is to build an artificial intelligence model capable of composing and producing music. With this model, we pretend to create audio that can be categorized in a music genre. Specifically, we use LoFi as the main genre category for our experiments. With this in mind, we try to generate LoFi music tracks, clearly differentiated from sound or noise, that sound as much similar as possible to human generated tracks.

The principal task of the project is to generate recurrently the next note of the track, or in other words, generate sequences. This gives us a first idea about which kind of neural network is needed to solve the task. In the same way, it is required to generate chords (groups of three or more notes that sound at the same time) and also to introduce as features silences and the durations of the notes.

Moreover, it is needed to apply some post-production process in order to tune the results with typical sounds of LoFi music such as a characteristic beat or the sound of the rain. Furthermore, it is desired to upload some of the generated tracks in a streaming platform in order to make it accesible to anyone and also to allow us to determine the evaluation of the results.

Our approach proposes a generation engine made of two recurrent neural networks. One is responsible for generating the melody, second for the chords.

## End to end generator

The project generates unique melody and chords which are mastered by adding a typical for Lofi beats and background noises. Both chords and melody can be played by any instrument provided in [sf2](https://en.wikipedia.org/wiki/SoundFont) format. Additionally the system allows to add an inifnite (reasonable) number of post effects like rain, beats etc. in a wav format. Check [How to add your post effects & instruments](#how-to-add-your-post-effects--instruments). 

The program uses the above DNNs to generate midi files with chords and melody. You can find both files in a data/midi directory. Then the midi files are changed to WAV format by [FluidSynth](https://pypi.org/project/pyFluidSynth/). You can find the melody and chords in a wav format in data/wav directory. Then the posteffects are added by [AudioSegment](https://audiosegment.readthedocs.io/en/latest/audiosegment.html) simply overlaying one track over another. 

## Prerequisites

* Python 3.7+
* pip
* venv

It is strongly recommended running the project in a [Virtual env](https://docs.python.org/3/tutorial/venv.html)

## Build & Run


Activate your venv and install all requirements.
 ```sh
  pip install -r requirements.txt
  ```
Once done, you can run the project. You have 3 modes to chooose:
1. Training Notes Model Mode. 
```sh
  python main.py train_notes
  ```
2. Training Chords Model Mode. 
```sh
  python main.py train_chords
  ```
3. Training Music Generation Mode. 
```sh
  python main.py generate_music
  ```

The 3rd mode will guide you through a generation process where you can choose a specific options of the newly generated song.
The final song will be available in data/final_results directory.

## How to add your post effects & instruments

All post effects must be in a WAV format. Just upload them to the app/effects directory. When running the program in a generation mode will be able to choose your effects.

All instruments must be in a sf2 format. Just upload them to the app/sf2 directory. When running the program in a generation mode you will be able to choose your instruments.








## Data & Data Processing

To train our models we have used:

* 300 midi files (Anime and Lo-Fi songs).
* 32 hours of music.
* 28 Lo-fi Chords.
* 110.630 Notes, durations and velocities.

It was quite difficult to manage this data as we didn't had experience with audio processing and not prior experience manipulating MIDI files.

MIDI  is a technical standard that describes a communications protocol, digital interface, and electrical connectors that connect a wide variety of electronic musical instruments, computers, and related audio devices for playing, editing, and recording music. MIDI files contains relevant information about the songs, not only the notes, but their duration, velocity, tempo, instrument, scale among much more.

### Preprocessing:
From the Midi files we

* Extracted notes, durations and velocities and tempo from midi files using music21.

* Normalised Duration and Velocities and Tempo accordingly to Lo-fi music genre.

* Create index vocabularies. 
* Converted sequences of notes into index sequences to feed the model.
* Create batches of sequences of indexes of size 100.
