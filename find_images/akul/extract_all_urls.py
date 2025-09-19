import pandas as pd
import argparse

def extract_urls(parquet_path, column_name, output_txt_path):
    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    # Make sure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the parquet file.")

    # Extract and write URLs
    df[column_name].dropna().astype(str).to_csv(output_txt_path, index=False, header=False)
    print(f"✅ Extracted {len(df)} rows from '{column_name}' to '{output_txt_path}'.")

if __name__ == "__main__":
    parquet_path = r"C:\Users\ferba\Downloads\laion"
    column_name = "URL"
    output_txt_path = r"all_urls.txt"
    extract_urls(parquet_path, column_name, output_txt_path)
