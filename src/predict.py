"""Predict the class of one image using a saved model."""
import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array, get_file
from tensorflow.keras.models import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--model", default="models/image_recognition_model.keras")
    parser.add_argument("--labels", default="models/class_names.json")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError("Trained model not found. Train the model first using src/train.py.")
    if not Path(args.labels).exists():
        raise FileNotFoundError("Class names file not found. Train the model first.")

    model = load_model(args.model)
    with open(args.labels, encoding="utf-8") as f:
        class_names = json.load(f)

    image = load_img(args.image, target_size=(224, 224))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    probabilities = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(probabilities))
    print(f"Predicted class: {class_names[index]}")
    print(f"Confidence: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
