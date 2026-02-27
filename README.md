# Rare Bird Spotting Image Classifier (Macaw vs Pigeon vs Squirrel)

This project builds an image classification system to identify rare macaws and distinguish them from common animals (pigeons and squirrels). Two supervised learning models were implemented and compared: **K-Nearest Neighbours (KNN)** and **Support Vector Machine (SVM)**. The goal is a practical classifier that could support birdwatchers and conservation monitoring.

## Dataset
I used 3 public datasets and selected ~100 images per class:
- Squirrel dataset (images.cv)
- Pigeon dataset (images.cv)
- Lear’s macaw dataset (images.cv)

Images were organized into class folders and renamed with labels using Python.

## Preprocessing
- Resized all images to **64×64** to control feature size and training cost.
- Converted images to **grayscale** for HOG extraction.
- Kept **RGB** copies for color features.
- Normalized RGB pixel values to **0–1** by dividing by 255.

## Feature Extraction
Each image is converted into a single feature vector made from:
1) **HOG features** (shape/edge/texture)
2) **Color features** (flattened normalized RGB values)

The final feature vector is created by concatenating HOG + color features.

## Models
### KNN
- Tested k values from **1 to 10** using cross-validation.
- Selected the k with the highest average CV accuracy.

### SVM
- Trained an SVM classifier on the same feature vectors.
- Compared performance against KNN using standard classification metrics.

## Evaluation
Models were evaluated using:
- Accuracy
- Precision / Recall / F1-score (classification report)

**Results:**
- KNN test accuracy ≈ **51.67%**
- SVM test accuracy ≈ **83.33%**
SVM performed best overall, especially for separating macaw and squirrel classes.

## Real-World Testing
The final SVM model was tested on **new unseen images** (not from the training set).  
Result: **12/15 correct (~80%)**. Misclassifications were mostly due to lighting, background clutter, or unusual pose/orientation.


**Future Improvements**

More diverse training data (lighting/background/angles)

Better color descriptors (histograms instead of raw flatten)

Hyperparameter tuning for SVM (C, gamma, kernel)

Try CNN / transfer learning for higher accuracy

## How to Run
1) Install dependencies:
```bash
pip install -r requirements.txt
Run the notebook in google colab
Open notebooks/PA33_14_IlamaranRaju_project2.ipynb and run all cells.


