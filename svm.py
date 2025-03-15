import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load datasets
train_file = "Training_Data.xlsx"
dev_file = "Development_Data.xlsx"
test_file = "Test_Data.xlsx"

df_train = pd.read_excel(train_file)
df_dev = pd.read_excel(dev_file)
df_test = pd.read_excel(test_file)

# Define features and target variable
X_train = df_train.drop(columns=["Trade Action"])  # Features
y_train = df_train["Trade Action"]  # Target variable

X_dev = df_dev.drop(columns=["Trade Action"])  # Features
y_dev = df_dev["Trade Action"]  # Target variable

X_test = df_test.drop(columns=["Trade Action"])  # Features
y_test = df_test["Trade Action"]  # Target variable

# Convert categorical target variable to numeric labels
label_mapping = {"Buy": 2, "Hold": 1, "Sell": 0}
y_train = y_train.map(label_mapping)
y_dev = y_dev.map(label_mapping)
y_test = y_test.map(label_mapping)

# Balance dataset using SMOTE (Better balancing for "Hold" class)
sm = SMOTE(random_state=42, sampling_strategy={1: int(1.7 * sum(y_train == 1))})
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

# Standardize the features (SVM performs better with standardized data)
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Feature Selection using Recursive Feature Elimination (RFE)
svc_rfe = SVC(kernel="linear")  # Linear kernel for initial feature selection
rfe = RFE(svc_rfe, n_features_to_select=int(0.8 * X_train_balanced.shape[1]))  # Keep top 80% of features
X_train_balanced = rfe.fit_transform(X_train_balanced, y_train_balanced)
X_dev = rfe.transform(X_dev)
X_test = rfe.transform(X_test)

# Define SVM with Hyperparameter Tuning
param_grid = {
    'svc__C': [0.1, 1, 10],  # Extended range of regularization
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Added sigmoid kernel
    'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],  # More gamma values for testing
}

# Pipeline for Scaling and SVM Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize Data
    ('svc', SVC(probability=True, random_state=42))  # Support Vector Classifier
])

grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Best model from grid search
best_svm = grid_search.best_estimator_

# Evaluate the model on Development Set
y_dev_pred = best_svm.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")

# Evaluate the model on Test Set
y_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Confusion Matrix & Report for Test Set
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(6,6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('SVM Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
