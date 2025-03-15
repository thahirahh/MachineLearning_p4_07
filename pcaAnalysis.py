import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Training Data for PCA
df_train = pd.read_excel("Training_Data_PrePCA.xlsx")

# Drop 'Date' before PCA
X_train = df_train.drop(columns=["Date", "Trade Action"])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Scree Plot (Explained Variance)
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot (PCA Explained Variance)")
plt.show()

# Determine the optimal number of components (e.g., 95% variance explained)
cumulative_variance = np.cumsum(explained_variance)
n_components = np.argmax(cumulative_variance >= 0.95) + 1  # Select first index where variance >= 95%

print(f"Optimal number of principal components: {n_components}")

# Apply PCA again with optimal number of components
pca_optimal = PCA(n_components=n_components)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Get feature importance (PCA Loadings)
loading_scores = pd.DataFrame(pca_optimal.components_, columns=X_train.columns, index=[f"PC{i+1}" for i in range(n_components)])

# Display top contributing features for the first two principal components
top_features_pc1 = loading_scores.loc["PC1"].abs().nlargest(5)
top_features_pc2 = loading_scores.loc["PC2"].abs().nlargest(5)

# Combine top features from both PCs to create final feature selection list
selected_features = list(set(top_features_pc1.index).union(set(top_features_pc2.index)))
print("\nSelected Features for Testing:")
print(selected_features)

# Save the selected features for later use
pd.DataFrame(selected_features, columns=["Selected Features"]).to_excel("PCA_Selected_Features.xlsx", index=False)

# Scatter Plot of PCA Components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_optimal[:, 0], X_pca_optimal[:, 1], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Data Projected onto First Two Principal Components")
plt.show()