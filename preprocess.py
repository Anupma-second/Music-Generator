"""
STEP 1: Preprocessing
----------------------
Reads all MIDI files in midi_songs/, extracts notes and chords using music21,
and saves the resulting sequence of "events" to a pickle file for training.
"""

import glob
import pickle
from music21 import converter, instrument, note, chord


def get_notes(midi_folder="midi_songs"):
    """
    Parse every MIDI file in midi_folder and return a flat list of
    note/chord events as strings.

    - A single note becomes its pitch name, e.g. "C4"
    - A chord becomes a dot-separated string of pitch IDs, e.g. "4.7.11"
    """
    notes = []
    midi_files = glob.glob(f"{midi_folder}/*.mid")

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid files found in '{midi_folder}/'. Add some MIDI files first."
        )

    print(f"Found {len(midi_files)} MIDI files. Parsing...")

    for i, file in enumerate(midi_files):
        print(f"  [{i+1}/{len(midi_files)}] Parsing {file}")
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"    Skipping {file}: {e}")
            continue

        # Try to split out instrument parts; fall back to flat notes
        parts = instrument.partitionByInstrument(midi)

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    print(f"Extracted {len(notes)} note/chord events total.")
    return notes


if __name__ == "__main__":
    notes = get_notes()

    with open("models/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    print("Saved extracted notes to models/notes.pkl")
    print("Sample of first 20 events:", notes[:20])
