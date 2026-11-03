# Pet Adoption Speed Prediction Using Multimodal Data

## 1. Project Description
This project involves predicting pet adoption speed using multimodal data which consist of images, text, and tabular attributes obtained from PetFinder dataset. A comparison is carried out among five different modelling approaches which include: Image only, Text only, Tabular only, Early Fusion, and Late Fusion. Logistic Regression is used for the single-modality models and Early Fusion. Late Fusion combines their predicted probabilities using learned weights.

## 2. Dataset
Dataset: [PetFinder Adoption Prediction (Kaggle)](https://www.kaggle.com/c/petfinder-adoption-prediction/data)

Original dataset contains 14,993 pets; in this case, dataset contains 14,640 pets after data pre-processing. The labelled dataset is split into training (70%), validation (15%), and testing (15%) sets. The test set is created from the labelled training data, not the official Kaggle test set.

## 3. Methods
- Image: Feature extraction using ResNet18.
- Text: Feature extraction using Multilingual BERT.
- Tabular: Feature preprocessing using scaling and one-hot encoding.
- Early Fusion: Combining features before classification.
- Late Fusion: Combining predictions through learned weighting.

## 4. Results
Late Fusion gives the best performance on the held-out validation set (1,098 samples).

### Validation metrics
- Accuracy: 0.4117
- Macro F1: 0.3121
- Weighted F1: 0.3901
- Quadratic Weighted Kappa (QWK): 0.3473

It is worth mentioning that the model shows difficulties in predicting AdoptionSpeed class 0.

### Test metrics
- Accuracy: 0.4053
- Macro F1: 0.3157
- Weighted F1: 0.3819
- Quadratic Weighted Kappa (QWK): 0.3506
- Log Loss: 1.3512

## 5. Reproduction and Execution Instructions

1) Get train.csv and train_images.zip files from Kaggle website.
2) Put them in MyDrive/PetFinder/ folder in Google Drive.
3) Open PetFinder_Project2_2-2.ipynb in Google Colab.
4) Use GPU if available.
5) Install Python libraries (for example, scikit-learn, PyTorch, torchvision, transformers, Pillow, joblib) if required.
6) Run notebook cells sequentially.

The notebook saves extracted features, trained models, and evaluation results in Google Drive.
