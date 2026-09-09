# Brain Tumor MRI Classification

Deep learning-based classification of brain MRI images into four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Model

A pretrained **ResNet34** was fine-tuned using ImageNet weights with image augmentation and hyperparameter tuning.

## Results

- **Test Accuracy:** 98.26%
- **Weighted F1 Score:** 98.26%
- **Validation Accuracy:** 97.61%

The evaluation uses a duplicate-aware grouped split to reduce data leakage from visually identical MRI images.

## Features

- CNN baseline and ResNet34 comparison
- Data preprocessing and augmentation
- Duplicate/leakage analysis
- Confusion matrix and error analysis
- Streamlit-based MRI prediction app

## Tech Stack

Python · PyTorch · Torchvision · Scikit-learn · OpenCV/PIL · Streamlit

## Project Structure

```text
brain-tumor-mri-classification/
├── app.py
├── src/
├── results/
└── .gitignore
