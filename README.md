# User Manual for Stock Prediction System
## 1. Environment Setup
### 1.1 Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using:

```
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn matplotlib statsmodels openpyxl
```

## 2. Data Processing
### 2.1 Running Data Cleaning & Preparation
This step processes the raw dataset, removes missing values, normalizes the data, and separates it into Training, Development, and Test sets.

```
python dataPreparation.py
```
Output Files:
Training_Data_PrePCA.xlsx
Development_Data_PrePCA.xlsx
Test_Data_PrePCA.xlsx

### 2.2 Applying PCA for Feature Selection
This step identifies the most important features using Principal Component Analysis (PCA) and saves them for further processing.

```
python pcaAnalysis.py
```
Output File:
PCA_Selected_Features.xlsx

### 2.3 Applying Selected Features to the Dataset
This script updates the Training, Development, and Test datasets to only include PCA-selected features.

```
python applyPCASelection.py
```
Updated Output Files:
Training_Data.xlsx
Development_Data.xlsx
Test_Data.xlsx

## 3. Training & Testing Models
You can now train and test your machine learning models. Run any of the following scripts:

Support Vector Machine (SVM):
```
python svm.py
```
Random Forest:
```
python randomforest.py
```
LightGBM:
```
python lightGBM.py
```
XGBoost:
```
python XGBoost.py
```

## 4. Understanding the Results
Each model outputs:
✅ Accuracy Score – Overall model performance.
✅ Confusion Matrix – True vs. predicted classifications.
✅ Classification Report – Precision, Recall, F1-score for each class.

## 5. Reproducing the Results
1️⃣ Run dataPreparation.py to clean and split the dataset.
2️⃣ Run pcaAnalysis.py to get important features.
3️⃣ Run applyPCASelection.py to update the dataset.
4️⃣ Run any ML model script (svm.py, randomforest.py, lightGBM.py, XGBoost.py) for training & testing.
5️⃣ Analyze accuracy and predictions from the outputs.
