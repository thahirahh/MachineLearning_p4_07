import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

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

# Standardize Features (Improves Stability)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Balance dataset using SMOTE (Ensures "Hold" class is better balanced)
sm = SMOTE(random_state=42, sampling_strategy={1: int(1.8 * sum(y_train == 1))})  # Increased balancing
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

# Feature Selection using Random Forest
feature_selector = RandomForestClassifier(n_estimators=200, random_state=42)
feature_selector.fit(X_train_balanced, y_train_balanced)
selector = SelectFromModel(feature_selector, threshold="mean", prefit=True)  # Select features with importance >= mean
X_train_balanced = selector.transform(X_train_balanced)
X_dev = selector.transform(X_dev)
X_test = selector.transform(X_test)

# Train Random Forest Model with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=1200,  # More trees for better learning
    max_depth=20,  # Deeper trees to capture more patterns
    min_samples_split=5,  # Increases stability by requiring more samples per split
    min_samples_leaf=3,  # Ensures balanced leaf nodes
    max_features="sqrt",  # Improves feature diversity
    class_weight="balanced",  # Adjusts weight for class imbalance
    random_state=42,
    n_jobs=-1  # Uses all available CPU cores for faster training
)
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on Development Set
y_dev_pred = rf_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")

# Evaluate on Test Set
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Confusion Matrix & Report for Test Set
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Feature Importance Plot
importances = rf_model.feature_importances_
feature_names = df_train.drop(columns=["Trade Action"]).columns[selector.get_support()]
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx][:10], importances[sorted_idx][:10])  # Show top 10 features
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()
