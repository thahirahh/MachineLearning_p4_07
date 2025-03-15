import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load training dataset
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

# Balance dataset using SMOTE
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.2,  # Stronger L2 Regularization
    reg_alpha=0.7,  # Stronger L1 Regularization
    learning_rate=0.05,  # Lower learning rate
    n_estimators=700,  # More trees
    max_depth=7,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight={0: 1, 1: 1.5, 2: 1}  # Give "Hold" class more weight
)


# Expanded hyperparameter tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.5],  # Minimum loss reduction required to make a further partition
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring="accuracy", cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Best model from grid search
best_xgb = grid_search.best_estimator_

# Evaluate the model on Development Set
y_dev_pred = best_xgb.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")

# Evaluate the model on Test Set
y_pred = best_xgb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Confusion Matrix & Report for Test Set
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
