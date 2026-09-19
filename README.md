# Face Recognition Under Partial Occlusion

### A Regional Feature Fusion Approach using ArcFace & SVM

## Overview

This project implements an occlusion-aware face recognition system using
MATLAB. The system combines facial information from multiple regions to
improve recognition when parts of the face are covered.

The recognition pipeline uses a pretrained **ArcFace w600k_r50 ONNX
model** for facial feature extraction and an **RBF-kernel Support Vector
Machine (SVM)** for identity classification.

## Pipeline

``` text
Input Image
     ↓
Face Detection
     ↓
Face Cropping
     ↓
112 × 112 Preprocessing
     ↓
Regional Feature Extraction
     ↓
ArcFace Feature Extraction
     ↓
512-D Regional Embeddings
     ↓
Weighted Feature Fusion
     ↓
L2 Normalization + Standardization
     ↓
PCA (~98% Variance)
     ↓
RBF-SVM Classification
     ↓
Identity Prediction
```

## 1. Dataset Preparation

The dataset was manually prepared using face images of two individuals.

The dataset is divided into:

``` text
dataset/
├── train/
│   ├── person1/
│   └── person2/
│
└── test/
    ├── person1/
    └── person2/
```

Training images are used to train the recognition classifier, while
separate testing images are used for evaluation.

## 2. Face Detection and Preprocessing

For each input image:

1.  The face is detected using the MATLAB Cascade Object Detector.
2.  The largest detected face is selected.
3.  The detected face is cropped.
4.  The face is resized to **112 × 112 pixels**.
5.  The image is converted from **RGB to BGR**.
6.  Pixel normalization is performed using:

``` text
(pixel - 127.5) / 127.5
```

## 3. ArcFace Feature Extraction

The project uses the pretrained:

``` text
ArcFace w600k_r50 ONNX
```

ArcFace converts the processed face image into a numerical facial
embedding.

Each embedding contains:

``` text
512 features
```

The pretrained network is used as a feature extractor rather than
training a deep face-recognition network from the beginning.

## 4. Regional Feature Extraction

To handle partial occlusion, features are extracted from multiple facial
regions:

-   Full face
-   Upper region
-   Middle region
-   Lower region
-   Left region
-   Right region

Each region produces its own **512-dimensional ArcFace representation**.

The purpose is to retain useful identity information from visible parts
when another region is occluded.

## 5. Weighted Feature Fusion

The regional embeddings are combined using weighted feature fusion.

``` text
Full Face
Upper Region
Middle Region
Lower Region
Left Region
Right Region
       ↓
Weighted Feature Fusion
       ↓
Combined Feature Representation
```

This allows information from visible facial regions to contribute to the
final representation.

## 6. Normalization and Standardization

After feature fusion:

-   L2 normalization is applied.
-   Standardization is performed.

The same preprocessing parameters obtained during training are applied
during testing so that the test features remain consistent with the
training features.

## 7. PCA Dimensionality Reduction

Principal Component Analysis (PCA) is applied to the fused
representation.

The implementation retains approximately:

``` text
98% of the original variance
```

PCA reduces redundant information and makes the subsequent
classification process more efficient.

## 8. RBF-SVM Classification

The PCA-reduced features are provided to an **RBF-kernel Support Vector
Machine**.

During training, the SVM learns the feature patterns associated with:

``` text
Person 1
Person 2
```

During testing, the same preprocessing and PCA transformation are
applied before the SVM predicts the identity.

## 9. Testing

The trained model is saved as:

``` text
ArcFaceRecognitionModel.mat
```

The `test.m` script loads the trained model and evaluates the system
using the separate test dataset.

For every test image, the pipeline performs:

``` text
Face Detection
      ↓
Regional Feature Extraction
      ↓
Feature Fusion
      ↓
Normalization
      ↓
Standardization
      ↓
PCA Transformation
      ↓
SVM Classification
```

## 10. Experimental Results

The system was evaluated using:

``` text
Total test images : 20
Person 1          : 10 images
Person 2          : 10 images
Correctly classified: 19
Incorrectly classified: 1
Overall accuracy   : 95%
```

Reported class-wise accuracy:

  Class        Accuracy
  ---------- ----------
  Person 1          90%
  Person 2         100%
  Overall           95%

## Project Structure

A suggested GitHub repository structure is:

``` text
Face-Recognition-Under-Partial-Occlusion/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── main.m
│   ├── train.m
│   ├── test.m
│   ├── faceDetection.m
│   ├── extractRegionalFeatures.m
│   └── featureFusion.m
│
├── models/
│   └── README.md
│
├── dataset/
│   ├── train/
│   │   ├── person1/
│   │   └── person2/
│   └── test/
│       ├── person1/
│       └── person2/
│
├── results/
│   ├── confusion_matrix/
│   └── sample_results/
│
├── docs/
│   └── project_report.pdf
│
└── requirements.md
```

> The filenames in `src/` above are suggested names. Replace them with
> the actual MATLAB filenames used in the implementation.

## Technologies Used

-   MATLAB
-   ArcFace
-   ONNX
-   RBF-SVM
-   PCA
-   Image Processing
-   Face Detection
-   Feature Fusion

## Key Features

-   Partial-occlusion-aware face recognition
-   Multi-region facial feature extraction
-   Pretrained ArcFace embeddings
-   512-dimensional facial representations
-   Weighted regional feature fusion
-   L2 normalization and standardization
-   PCA dimensionality reduction
-   RBF-SVM classification
-   Separate training and testing datasets

## Result

The reported experiment achieved **95% overall test accuracy** on 20
test images, with 19 correctly recognized images and 1 incorrectly
classified image.

## Notes

The dataset contains images of two individuals and should only be
published when appropriate consent and usage rights are available.

The pretrained ArcFace model should also be distributed only in
accordance with its applicable model/license terms. If the model is not
included in the repository, provide instructions for obtaining it in
`models/README.md`.

## Project Report

The detailed project technical report is available in the `docs/`
directory.

------------------------------------------------------------------------

**Project:** Face Recognition Under Partial Occlusion\
**Approach:** Regional Feature Fusion using ArcFace & RBF-SVM\
**Platform:** MATLAB
