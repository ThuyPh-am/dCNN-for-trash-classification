# src/model.py

from tensorflow.keras import layers, models


def build_model(num_classes, img_size=(256, 256)):
    """
    Build and compile CNN model for image classification.

    Args:
        num_classes (int): Number of output classes
        img_size (tuple): Image size (height, width)

    Returns:
        model (tf.keras.Model): Compiled model
    """

    model = models.Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),

        # Normalize pixel values
        layers.Rescaling(1./255),

        # Feature extraction
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # Classification head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model