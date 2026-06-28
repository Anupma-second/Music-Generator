"""
STEP 3: Generation
-------------------
Loads the trained model + vocabulary mapping, picks a random seed sequence,
predicts new notes one at a time (feeding each prediction back in as input),
and converts the resulting sequence into a playable MIDI file.
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream

SEQUENCE_LENGTH = 50


def generate_notes(model, network_input, pitch_names, n_vocab, num_notes=200, temperature=1.0):
    """
    Generate a new sequence of notes.

    network_input: the full set of training input sequences (we pick a random one as a seed)
    temperature: >1.0 = more random/creative, <1.0 = more conservative/repetitive
    """
    int_to_pitch = {i: p for i, p in enumerate(pitch_names)}

    # pick a random starting sequence from the training data as our "seed"
    start = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start].flatten())  # already normalized 0-1 values

    prediction_output = []

    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(prediction_input, verbose=0)[0]

        # temperature sampling for variety (instead of always picking argmax)
        prediction = np.log(prediction + 1e-9) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.random.choice(len(prediction), p=prediction)

        result = int_to_pitch[index]
        prediction_output.append(result)

        # slide the window forward: drop first note, append the new (normalized) one
        pattern.append(index / float(n_vocab))
        pattern = pattern[1:]

    return prediction_output


def create_midi(prediction_output, output_path="output/generated_music.mid"):
    """Convert a list of note/chord strings back into a music21 Stream and save as MIDI."""
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # chord pattern, e.g. "4.7.11"
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # single note, e.g. "C4"
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5  # space notes half a beat apart

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Generated MIDI saved to {output_path}")


if __name__ == "__main__":
    # 1. Load vocabulary mapping saved during training
    with open("models/pitch_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    pitch_names = mapping["pitch_names"]
    n_vocab = len(pitch_names)

    # 2. Load notes again to rebuild the same network_input shape used for training
    with open("models/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    pitch_to_int = mapping["pitch_to_int"]
    network_input = []
    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i : i + SEQUENCE_LENGTH]
        network_input.append([pitch_to_int[p] for p in seq_in])
    network_input = np.reshape(network_input, (len(network_input), SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)

    # 3. Load trained model
    model = load_model("models/final_model.keras")
    print("Model loaded.")

    # 4. Generate new notes
    generated = generate_notes(model, network_input, pitch_names, n_vocab, num_notes=150, temperature=1.0)
    print("Generated sequence sample:", generated[:20])

    # 5. Save as MIDI
    create_midi(generated, output_path="output/generated_music.mid")
