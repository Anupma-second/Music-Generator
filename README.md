# AI Music Generator

An AI-powered music generation system that learns musical patterns from existing MIDI files and composes new note sequences using a deep learning (LSTM) model.

## Overview

This project takes a folder of MIDI files, learns the patterns of notes and chords using an LSTM (Long Short-Term Memory) neural network, and then generates brand new music based on what it learned, saved as a playable .mid file.

Pipeline: MIDI files -> Preprocessing -> LSTM Training -> Music Generation -> New MIDI file

## Tech Stack

- Python 3
- music21 - parsing MIDI files, extracting notes/chords, writing generated output back to MIDI
- TensorFlow / Keras - building and training the LSTM model
- NumPy - data reshaping and normalization

## Project Structure

music-generator-ai/
- preprocess.py     Step 1: extract note/chord sequences from MIDI files
- train.py           Step 2: build and train the LSTM model
- generate.py         Step 3: generate new music and save as MIDI
- midi_songs/         (not committed) training MIDI files go here
- models/              (not committed) trained model and vocabulary mapping
- output/               (not committed) generated MIDI files

Note: midi_songs/, models/, and output/ contents are excluded via .gitignore since they are large binary/generated files.

## How It Works

### 1. Preprocessing (preprocess.py)
Parses every .mid file in midi_songs/ using music21. Each note becomes its pitch name, and each chord becomes a dot-separated string of pitch IDs. The full sequence of events is saved to models/notes.pkl.

### 2. Training (train.py)
Builds a vocabulary of all unique notes/chords, splits the note sequence into overlapping windows of 50 notes to predict the next note, trains a 2-layer LSTM network (256 units each, with dropout), and saves the trained model and vocabulary mapping.

### 3. Generation (generate.py)
Loads the trained model and vocabulary, picks a random seed sequence, repeatedly predicts the next note using temperature sampling for variety, and writes a new timestamped MIDI file to output/.

## Setup and Usage

### 1. Install dependencies
pip install music21 tensorflow

### 2. Add training data
Place .mid files directly inside midi_songs/.

### 3. Run the pipeline in order
python preprocess.py
python train.py
python generate.py

### 4. Listen to the result
Open the generated .mid file in any media player, DAW, or online MIDI player.

## Notes and Possible Improvements

- More training data and more training epochs generally produce more musically coherent results.
- The temperature parameter in generate.py controls randomness.
- Possible extensions: a simple web UI, multi-instrument generation, or converting MIDI output to audio.

## Acknowledgements

Built as part of an AI/ML internship task on Music Generation with AI.
