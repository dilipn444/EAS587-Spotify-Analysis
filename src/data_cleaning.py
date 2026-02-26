import pandas as pd
import numpy as np
import os

def clean_data():
    # Define paths
    raw_path = 'project_repo/data/raw/raw_spotify_tracks.csv'
    processed_path = 'project_repo/data/processed/cleaned_spotify_tracks.csv'

    print("--- Starting Data Cleaning ---")
    
    # Check if file exists
    if not os.path.exists(raw_path):
        # Fallback for local execution if running outside Colab later
        raw_path = '../data/raw/raw_spotify_tracks.csv'
        processed_path = '../data/processed/cleaned_spotify_tracks.csv'

    df = pd.read_csv(raw_path)
    print(f"Initial Shape: {df.shape}")

    # --- 10 DISTINCT OPERATIONS ---

    # Op 1: Remove Duplicates (Data Integrity)
    # Check for duplicate track IDs
    if 'track_id' in df.columns:
        df = df.drop_duplicates(subset=['track_id'])
    else:
        df = df.drop_duplicates()
    print("1. Dropped duplicates.")

    # Op 2: Handle Missing Values (Cleaning)
    # Drop rows where essential info (name/artist) is missing
    cols_to_check = [col for col in ['track_name', 'artists'] if col in df.columns]
    df = df.dropna(subset=cols_to_check)
    print("2. Dropped rows with missing text data.")

    # Op 3: Rename Columns (Standardization)
    # Rename 'duration_ms' to be more descriptive if it exists
    df.rename(columns={'duration_ms': 'duration_ms_raw', 'explicit': 'is_explicit'}, inplace=True)
    print("3. Renamed columns.")

    # Op 4: Unit Conversion (Feature Engineering)
    # Convert ms to minutes for better readability
    if 'duration_ms_raw' in df.columns:
        df['duration_mins'] = df['duration_ms_raw'] / 60000
    print("4. Converted duration to minutes.")

    # Op 5: Boolean to Integer (Type Conversion)
    # Convert explicit True/False to 1/0
    if 'is_explicit' in df.columns:
        df['is_explicit'] = df['is_explicit'].astype(int)
    print("5. Converted explicit boolean to int.")

    # Op 6: String Normalization (Text Cleaning)
    # Lowercase track names to standardize
    if 'track_name' in df.columns:
        df['track_name'] = df['track_name'].str.lower().str.strip()
    print("6. Normalized track names.")

    # Op 7: Outlier Removal (Filtering)
    # Remove songs that are too short (< 30 seconds)
    if 'duration_mins' in df.columns:
        df = df[df['duration_mins'] >= 0.5]
    print("7. Removed short tracks (outliers).")

    # Op 8: Binning (Categorization)
    # Create popularity tiers: Low, Medium, High
    if 'popularity' in df.columns:
        df['popularity_tier'] = pd.cut(df['popularity'], bins=[-1, 33, 66, 100], labels=['Low', 'Medium', 'High'])
    print("8. Binned popularity.")

    # Op 9: Interaction Feature (Advanced Engineering)
    # Energy to Loudness Ratio
    if 'energy' in df.columns and 'loudness' in df.columns:
        # Avoid division by zero
        df['energy_loudness_ratio'] = df['energy'] / (df['loudness'].abs() + 0.001)
    print("9. Created energy/loudness ratio.")

    # Op 10: One-Hot Encoding (Preprocessing)
    # Encode 'time_signature' if it exists
    if 'time_signature' in df.columns:
        df = pd.get_dummies(df, columns=['time_signature'], prefix='ts')
    print("10. One-hot encoded time signature.")

    # --- SAVE ---
    df.to_csv(processed_path, index=False)
    print(f"✅ Cleaning Complete. Final Shape: {df.shape}")
    print(f"Saved to: {processed_path}")

if __name__ == "__main__":
    clean_data()
