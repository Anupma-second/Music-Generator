"""
STEP 2: Training
-----------------
Loads the extracted notes from preprocess.py, converts them into numeric
sequences, builds an LSTM model, and trains it to predict the next note
given a sequence of previous notes.
"""

import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

SEQUENCE_LENGTH = 50  # how many previous notes the model looks at to predict the next one


def prepare_sequences(notes, n_vocab):
    """
    Convert the list of note-strings into:
      - network_input:  normalized integer sequences  (X)
      - network_output: one-hot encoded next-note      (y)
      - pitch_to_int: mapping dict (saved for generation later)
    """
    pitch_names = sorted(set(notes))
    pitch_to_int = {pitch: i for i, pitch in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i : i + SEQUENCE_LENGTH]
        seq_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([pitch_to_int[p] for p in seq_in])
        network_output.append(pitch_to_int[seq_out])

    n_patterns = len(network_input)

    # reshape to (samples, timesteps, features) and normalize to 0-1 for LSTM
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)

    # one-hot encode the output
    network_output = np.eye(n_vocab)[network_output]

    return network_input, network_output, pitch_to_int


def build_model(input_shape, n_vocab):
    """LSTM network: this is the 'deep learning model' the task asks for."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


if __name__ == "__main__":
    # 1. Load preprocessed notes
    with open("models/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    pitch_names = sorted(set(notes))
    n_vocab = len(pitch_names)
    print(f"Total notes: {len(notes)} | Unique vocabulary size: {n_vocab}")

    if len(notes) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough notes ({len(notes)}) for SEQUENCE_LENGTH={SEQUENCE_LENGTH}. "
            "Add more MIDI files or reduce SEQUENCE_LENGTH."
        )

    # 2. Prepare training sequences
    network_input, network_output, pitch_to_int = prepare_sequences(notes, n_vocab)
    print("Training input shape:", network_input.shape)
    print("Training output shape:", network_output.shape)

    # save the vocabulary mapping — needed later for generation
    with open("models/pitch_mapping.pkl", "wb") as f:
        pickle.dump({"pitch_to_int": pitch_to_int, "pitch_names": pitch_names}, f)

    # 3. Build model
    model = build_model((network_input.shape[1], network_input.shape[2]), n_vocab)
    model.summary()

    # 4. Train
    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )

    # stop automatically once loss stops improving, so you don't wait for all
    # EPOCHS if the model has already converged
    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    EPOCHS = 50       # increase for better results with more data
    BATCH_SIZE = 128  # larger batch = fewer steps per epoch = faster on CPU with this much data

    model.fit(
        network_input,
        network_output,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
    )

    # 5. Save final model
    model.save("models/final_model.keras")
    print("Training complete. Model saved to models/final_model.keras")