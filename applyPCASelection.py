import pandas as pd

# Load selected features from PCA
selected_features = pd.read_excel("PCA_Selected_Features.xlsx")["Selected Features"].tolist()

# Load datasets
df_train = pd.read_excel("Training_Data_PrePCA.xlsx")
df_dev = pd.read_excel("Development_Data_PrePCA.xlsx")
df_test = pd.read_excel("Test_Data_PrePCA.xlsx")

# Keep only selected PCA features
df_train = df_train[selected_features + ["Trade Action"]]
df_dev = df_dev[selected_features + ["Trade Action"]]
df_test = df_test[selected_features + ["Trade Action"]]

# Save final datasets for model training
df_train.to_excel("Training_Data.xlsx", index=False)
df_dev.to_excel("Development_Data.xlsx", index=False)
df_test.to_excel("Test_Data.xlsx", index=False)

print("Final dataset with PCA-selected features and Trade Action saved!")