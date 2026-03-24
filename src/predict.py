# src/predict.py

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (256, 256)
MODEL_PATH = "models/trash_classifier.keras"

# Keep consistent with training
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def load_model():
    """Load trained model"""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_path):
    """Load and preprocess image"""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_path):
    """Run inference on a single image"""
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array)
    probs = tf.nn.softmax(predictions[0]).numpy()

    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = np.max(probs)

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Trash classification inference")
    parser.add_argument("image_path", type=str, help="Path to input image")

    args = parser.parse_args()

    model = load_model()
    predict_image(model, args.image_path)


if __name__ == "__main__":
    main()