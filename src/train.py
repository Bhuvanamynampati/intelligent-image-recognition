"""Train an image classifier using MobileNetV2 transfer learning.

Expected structure:
data_dir/
  train/class_name/*.jpg
  validation/class_name/*.jpg
  test/class_name/*.jpg
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def make_dataset(path, image_size, batch_size, training=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=training,
        seed=42,
    )
    class_names = ds.class_names
    if training:
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
        ])
        ds = ds.map(
            lambda x, y: (preprocess_input(augmentation(tf.cast(x, tf.float32), training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE), class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset", help="Folder containing train/validation/test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    root = Path(args.data_dir)
    train_path, val_path, test_path = root / "train", root / "validation", root / "test"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}. See README.md for the expected dataset structure.")

    image_size = (args.image_size, args.image_size)
    train_ds, class_names = make_dataset(train_path, image_size, args.batch_size, training=True)
    val_ds, val_names = make_dataset(val_path, image_size, args.batch_size)
    test_ds, test_names = make_dataset(test_path, image_size, args.batch_size)
    if class_names != val_names or class_names != test_names:
        raise ValueError("Train, validation, and test folders must have matching class names.")

    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Add images for at least two classes to dataset/train, validation, and test.")

    base_model = MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(args.image_size, args.image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    model.save("models/image_recognition_model.keras")
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=range(num_classes),
                                target_names=class_names, zero_division=0))

    with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=range(num_classes),
                                      target_names=class_names, zero_division=0))
        f.write(f"\nTest loss: {test_loss:.4f}\nTest accuracy: {test_accuracy:.4f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/accuracy.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss.png", dpi=150)
    plt.close()

    print("\nSaved model to models/image_recognition_model.keras")
    print("Saved graphs and report in outputs/")


if __name__ == "__main__":
    main()
