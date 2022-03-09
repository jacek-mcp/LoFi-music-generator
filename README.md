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
  * [END TO END SYSTEM](#end-to-end-system)
  * [Prerequisites](#prerequisites)
  * [Build & Run](#build-&-run)
  * [How to add your post effects & instruments](#how-to-add-your-post-effects-&-instruments)
	 

## About The Project
lorem ipsum blah blah blah

The generation engine is made of two recurrent neural networks. One is responsible for generating the melody, second for the chords.

blah blah blah

## Prerequisites

* Python 3.7
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






