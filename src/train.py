# src/train.py

import os
import tensorflow as tf

from data_loader import load_data
from model import build_model


DATA_DIR = "data/garbage"
MODEL_PATH = "models/trash_classifier.keras"
EPOCHS = 10


def main():

    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists. Skipping training.")
        return


    train_ds, val_ds, class_names = load_data(DATA_DIR)

    num_classes = len(class_names)
    print("Classes:", class_names)


    model = build_model(num_classes)


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )


    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()