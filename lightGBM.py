import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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

# Standardize Features (Improves LightGBM Performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Balance dataset using SMOTE (Better balance for "Hold" class)
sm = SMOTE(random_state=42, sampling_strategy={1: int(1.5 * sum(y_train == 1))})  # Increased balance
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

# Define LightGBM model with further optimized hyperparameters
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    learning_rate=0.035,  # Slightly increased for better convergence
    max_depth=9,  # Deeper trees allow better learning
    n_estimators=1400,  # More trees to strengthen decision boundaries
    num_leaves=50,  # More leaf nodes for better flexibility
    subsample=0.82,  # Prevents overfitting while keeping diversity
    colsample_bytree=0.88,  # Better feature selection
    reg_lambda=1.4,  # L2 Regularization (slightly lower to prevent restriction)
    reg_alpha=1.0,  # L1 Regularization (slightly adjusted)
    scale_pos_weight=1.8,  # Improves "Hold" class recall
)

# Train the LightGBM model
lgb_model.fit(
    X_train_balanced, y_train_balanced, 
    eval_set=[(X_dev, y_dev)], 
    callbacks=[lgb.early_stopping(50)]
)

# Evaluate on Development Set
y_dev_pred = lgb_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")

# Evaluate on Test Set
y_pred = lgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Confusion Matrix & Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Feature Importance Plot with Actual Column Names
feature_names = df_train.drop(columns=["Trade Action"]).columns  # Extract feature names

lgb.plot_importance(lgb_model, max_num_features=10)  # Limits to top 10 features
plt.xticks(rotation=45)  # Rotate labels for better visibility
plt.yticks(range(len(feature_names)), feature_names)  # Use actual feature names
plt.show()