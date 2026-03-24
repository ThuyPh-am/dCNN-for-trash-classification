# src/data_loader.py

import tensorflow as tf


def load_data(data_dir, img_size=(256, 256), batch_size=32):
    """
    Load training and validation datasets from directory.

    Args:
        data_dir (str): Path to dataset directory
        img_size (tuple): Image size (height, width)
        batch_size (int): Batch size

    Returns:
        train_ds, val_ds: TensorFlow datasets
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"   
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names