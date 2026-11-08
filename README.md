# Intelligent Image Recognition System

An image-classification project using Python, TensorFlow/Keras, and MobileNetV2 transfer learning.

## Project status
This is a starter repository. Add your chosen image dataset, run the training code, and replace the example results in your report with the real results from your run.

## Features
- Load images from class-named folders
- Resize images to 224 × 224
- Apply MobileNetV2 preprocessing and training-only augmentation
- Train a MobileNetV2 transfer-learning classifier
- Evaluate test accuracy and print precision, recall, and F1-score
- Save the trained model and training graphs
- Predict the class of a new image

## Folder structure

```text
intelligent-image-recognition/
├── dataset/
│   ├── train/
│   │   ├── class_a/
│   │   └── class_b/
│   ├── validation/
│   │   ├── class_a/
│   │   └── class_b/
│   └── test/
│       ├── class_a/
│       └── class_b/
├── models/                  # generated model files (not committed)
├── notebooks/
│   └── Intelligent_Image_Recognition.ipynb
├── outputs/                 # generated graphs/reports (not committed)
├── src/
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

Replace `class_a` and `class_b` with your actual class names. Each split should contain the same class folders. The starter training script supports two or more classes.

## Dataset split
A suggested split is approximately 70% train, 15% validation, and 15% test. Keep test images separate from training and validation. Avoid placing duplicate images across splits.

## Run in Google Colab (recommended for beginners)
1. Open https://colab.research.google.com/
2. Upload `notebooks/Intelligent_Image_Recognition.ipynb` or open the notebook from your GitHub repository.
3. Upload your dataset as a ZIP file and extract it to `/content/dataset`, or clone/download this repository and upload the dataset separately.
4. Run the notebook cells in order.
5. Check the printed metrics and generated graphs. Do not invent accuracy values; report the values from your actual run.

## Run locally
Python 3.10 or 3.11 is recommended for a beginner setup. Create a virtual environment, install dependencies, and place your dataset in the folders shown above.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Train:
```bash
python src/train.py --data_dir dataset --epochs 5
```

Predict a new image:
```bash
python src/predict.py --image path/to/your_image.jpg
```

Deep-learning installation can vary by operating system. If local TensorFlow setup is difficult, use Google Colab as described above.

## Technologies
- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## Important notes
- The repository does not include a dataset or a trained model.
- Do not upload private, licensed, or very large datasets unless you have permission.
- Model files and generated outputs are ignored by Git by default.
- Add your internship-specific requirements and supervisor-approved dataset before submission.
