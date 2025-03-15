import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "APPLtechindicator.xlsx"
df = pd.read_excel(file_path)

# Remove rows with null values
df_cleaned = df.dropna().copy()

# Format 'Date' column to remove time if it exists
if 'Date' in df_cleaned.columns:
    df_cleaned.loc[:, 'Date'] = pd.to_datetime(df_cleaned['Date']).dt.strftime('%Y-%m-%d')

# Define trading labels (Buy, Hold, Sell) based on price movement
X_percent = 2  # Define percentage threshold for Buy/Sell
N_days = 5  # Define the number of days to look ahead

def classify_trade_action(prices, X_percent, N_days):
    labels = []
    for i in range(len(prices) - N_days):
        future_return = ((prices[i + N_days] - prices[i]) / prices[i]) * 100
        if future_return > X_percent:
            labels.append("Buy")
        elif future_return < -X_percent:
            labels.append("Sell")
        else:
            labels.append("Hold")
    labels += ["Hold"] * N_days  # Fill last N days with "Hold"
    return labels

# Apply the rule to the last price column
df_cleaned.loc[:, 'Trade Action'] = classify_trade_action(df_cleaned['Last Price'].values, X_percent, N_days)

# Ensure no NaN values remain in 'Trade Action'
df_cleaned.loc[:, 'Trade Action'] = df_cleaned['Trade Action'].fillna("Hold")

# Select only numeric features for processing (PCA will decide best ones later)
df_numeric = df_cleaned.select_dtypes(include=[np.number])

# Normalize all features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# Add back 'Date' and 'Trade Action' columns
df_scaled["Date"] = df_cleaned["Date"]
df_scaled["Trade Action"] = df_cleaned["Trade Action"]

# Ensure there are no NaNs before splitting
if df_scaled.isnull().sum().sum() > 0:
    print("Warning: NaN values detected after preprocessing.")
    df_scaled = df_scaled.dropna()  # Drop any remaining NaNs just in case

# Split data into training (70%), development (15%), and test (15%)
X_train, X_temp = train_test_split(df_scaled, test_size=0.30, random_state=42, stratify=df_scaled["Trade Action"])
X_dev, X_test = train_test_split(X_temp, test_size=0.50, random_state=42, stratify=X_temp["Trade Action"])

# Save pre-PCA datasets
X_train.to_excel("Training_Data_PrePCA.xlsx", index=False)
X_dev.to_excel("Development_Data_PrePCA.xlsx", index=False)
X_test.to_excel("Test_Data_PrePCA.xlsx", index=False)

print("Data cleaned & separated. Now proceed with PCA analysis.")
