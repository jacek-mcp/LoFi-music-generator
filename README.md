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


## Data & Data Processing

To train our models we have used:

* 300 midi files (Anime and Lo-Fi songs).
* 32 hours of music.
* 28 Lo-fi Chords.
* 110.630 Notes, durations and velocities.

It was quite difficult to manage this data as we didn't had experience with audio processing and not prior experience manipulating MIDI files.

MIDI  is a technical standard that describes a communications protocol, digital interface, and electrical connectors that connect a wide variety of electronic musical instruments, computers, and related audio devices for playing, editing, and recording music. MIDI files contains relevant information about the songs, not only the notes, but their duration, velocity, tempo, instrument, scale among much more.

![image](https://github.com/jacek-mcp/LoFi-music-generator/blob/main/Screenshot%202022-03-10%20at%2001.48.20.png?raw=true)

### Preprocessing:
From the Midi files we

* Extracted notes, durations and velocities and tempo from midi files using music21.

* Normalised Duration and Velocities and Tempo accordingly to Lo-fi music genre.

* Create index vocabularies. 
* Converted sequences of notes into index sequences to feed the model.
* Create batches of sequences of indexes of size 100.


## Hypothesis
The original idea was to create a LSTM model that predicts sequences of not only notes but notes and duration. However, to add more expression to the songs, we decided to also include Velocity. That was a challenge since we weren't sure how we could integrate that the the current model.

## Arquitectures

* 1.The most straightforward approach was to have a huge index vocabulary of each note, its duration and its velocity, which is the first image on the left.
The result was gigant vocabulary, almost 10000 indexes, this could make the model heavy but simple, not challenging. There are other cons that we would explain later on the slides.

* 2.The next approach we took following our intuition was to add complexity by joining notes and durations into one single index vocabulary and having an isolated vocabulary for velocities, with two corresponding embedding layers. 2nd graph

* 3.After reviewing the results, we decided go further and try with one vocabulary by feature, therefore, 3. 3rd graph
Also trained an independent Chord model. Which results are concatenated to the main model predictions at the postproduction process adjusting the temp, the time signature and so on.

![image](https://github.com/jacek-mcp/LoFi-music-generator/blob/main/Screenshot%202022-03-10%20at%2009.48.34.png?raw=true)

## Results
As you might imagine this is not a typical classification task, we want to generate new improvised songs based on big training date of complex songs. To avoid the model predicting the exact same songs that have been trained for, we need to assess different losses and see how high is their overfitting in terms of predicting original songs and the quality of the predictions.

A high loss might result in absence of overfitting but produce extremely disonant results or repetitive sequences. On the other hand, a low loss might result in good results but very similar or identical to original songs, therefore the effort here is to select a loss that predicts good sounds while maintaining a low overfitting line.

From the graph we observe that the model 2 is generating more repetitive songs. 80-90% of the sequences notes matches the previous sequence. While the model 1 and 3, predicts less repetitive sequences.


In the following graph we can see the quality of each model's prediction.
x axis: Predicted song timestep
y axis: percentage of matching notes within the current sequence and the previous. (x-10:x) vs (x-20:x-10)

![image](https://github.com/jacek-mcp/LoFi-music-generator/blob/main/Screenshot%202022-03-10%20at%2002.10.05.png?raw=true)


The single embeddding's low repetitiveness is related to the fact that the model tends to predict exactly the same  fragments of the training data instead of improvising, overfitting.

See green line, around 30% of the predicted sequences higher than 15 notes, matches sequences of the training data.

While the two embeddings and three embeddings overfitting lines stay flatten.

![image](https://github.com/jacek-mcp/LoFi-music-generator/blob/main/Screenshot%202022-03-10%20at%2002.11.53.png?raw=true)


## Conclusions

* Take out 1: Model 3 and 1 predicts less repetitive songs
* Take out 2: Model 3 predicts predicts more original songs (less overfitting)

1 Embedding model: A big vocabulary may not impact the model in terms of predicting different notes, since you have one unique embedding for all three features, it's easier for the model to overfit and predict the right sequence.

2 embeddings model: with medium size vocabulary of notes and durations, the diversity of notes and durations is higher, as you have an index vocab of repeated notes with different durations, and therefore the model might find it harder to understand the next note. (ends up predicting one sequence over and over.)

3 embeddings model: One small vocabulary for each feature proves a better result, seems to make the model hard to get repetitive as there is a diverse unique vocabulary of only notes, therefore the model have more clear decisioning on which note comes next, as there are fewer unique notes.

## End to end generator

The project generates unique melody and chords which are mastered by adding a typical for Lofi beats and background noises. Both chords and melody can be played by any instrument provided in [sf2](https://en.wikipedia.org/wiki/SoundFont) format. Additionally the system allows to add an inifnite (reasonable) number of post effects like rain, beats etc. in a wav format. Check [How to add your post effects & instruments](#how-to-add-your-post-effects--instruments). 

The program uses the below DNNs to generate midi files with chords and melody. You can find both files in a data/midi directory. Then the midi files are changed to WAV format by [FluidSynth](https://pypi.org/project/pyFluidSynth/). You can find the melody and chords in a wav format in data/wav directory. Then the posteffects are added by [AudioSegment](https://audiosegment.readthedocs.io/en/latest/audiosegment.html) simply overlaying one track over another. 


![image](https://github.com/jacek-mcp/LoFi-music-generator/blob/main/end-to-end-flow.drawio.png?raw=true)

## Prerequisites

* Python 3.7+
* pip
* venv
* [FluidSynth](https://www.fluidsynth.org/)

It is strongly recommended running the project in a [Virtual env](https://docs.python.org/3/tutorial/venv.html)

### FluidSynth installation

Debian based linux
 ```sh
  sudo apt update
  sudo apt install fluidsynth
  ```
MacOS
 ```sh
  brew install fluid-synth
  ```


## Build & Run


Activate your venv and install all requirements.
 ```sh
  pip install -r requirements.txt
  ```
Once done, run the project using the below command in the project's root directory.

```sh
  python -m app.main generate_music
  ```

The 3rd mode will guide you through a generation process where you can choose a specific options of the newly generated song.
The final song will be available in data/final_results directory.

## How to add your post effects & instruments

All post effects must be in a WAV format. Just upload them to the app/effects directory. When running the program in a generation mode will be able to choose your effects.

All instruments must be in a sf2 format. Just upload them to the app/sf2 directory. When running the program in a generation mode you will be able to choose your instruments.







