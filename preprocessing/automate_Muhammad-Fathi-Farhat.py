# automate_Muhammad-Fathi-Farhat.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import os

def preprocess_data(input_url, output_path):
    # Load data langsung dari URL GitHub
    df = pd.read_csv(input_url)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    # Drop 'CustomerID' if exists
    if 'CustomerID' in df.columns:
        df_preprocessed = df.drop('CustomerID', axis=1)
    else:
        df_preprocessed = df.copy()

    # Scale numeric columns (excluding target 'Churn')
    columns_to_scale = [col for col in df_preprocessed.columns if col != 'Churn']
    scaler = StandardScaler()
    df_preprocessed[columns_to_scale] = scaler.fit_transform(df_preprocessed[columns_to_scale])

    # Save preprocessed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")


if __name__ == "__main__":
    input_url = "https://raw.githubusercontent.com/farhatfathi/Eksperimen_SML_Muhammad-Fathi-Farhat/refs/heads/main/Dataset_raw.csv"
    output_file = "preprocessing/churn_preprocessed.csv"
    preprocess_data(input_url, output_file)
