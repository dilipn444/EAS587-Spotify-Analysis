import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():

    # Validate input file path before processing
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # Load dataset and ensure output directory exists
    df = pd.read_csv(INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Apply consistent visualization theme
    sns.set_theme(style="whitegrid")

    # 1. Generate descriptive statistics summary
    desc_stats = df.describe()
    desc_stats.to_csv(f"{OUTPUT_DIR}/1_descriptive_statistics.csv")
    print("Saved descriptive statistics.")

    # 2. Create correlation heatmap for numeric features
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_correlation_heatmap.png")
    plt.close()
    print("Saved correlation heatmap.")

    # 3. Plot popularity distribution to inspect skewness
    if 'popularity' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['popularity'], bins=30, kde=True, color='purple')
        plt.title("Distribution of Track Popularity")
        plt.savefig(f"{OUTPUT_DIR}/3_popularity_distribution.png")
        plt.close()
        print("Saved popularity distribution.")

    # 4. Scatter plot: Danceability vs Energy (sampled for performance)
    if 'danceability' in df.columns and 'energy' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df.sample(min(1000, len(df))),
            x='danceability',
            y='energy',
            alpha=0.5
        )
        plt.title("Danceability vs Energy (Sampled)")
        plt.savefig(f"{OUTPUT_DIR}/4_dance_vs_energy_scatter.png")
        plt.close()
        print("Saved scatter plot.")

    # 5. Boxplot: Duration grouped by popularity tier
    if 'popularity_tier' in df.columns and 'duration_mins' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x='popularity_tier',
            y='duration_mins',
            data=df,
            palette="Set2"
        )
        plt.title("Song Duration by Popularity Tier")
        plt.savefig(f"{OUTPUT_DIR}/5_duration_boxplot.png")
        plt.close()
        print("Saved duration boxplot.")

    # 6. Count plot for explicit vs non-explicit tracks
    if 'is_explicit' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='is_explicit', data=df)
        plt.title("Count of Explicit vs Non-Explicit Tracks")
        plt.savefig(f"{OUTPUT_DIR}/6_explicit_count.png")
        plt.close()
        print("Saved explicit content count plot.")

    # 7. Violin plot for loudness distribution analysis
    if 'loudness' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df['loudness'], color='orange')
        plt.title("Violin Plot of Loudness")
        plt.savefig(f"{OUTPUT_DIR}/7_loudness_violin.png")
        plt.close()
        print("Saved loudness violin plot.")

    # 8. Time signature frequency analysis (one-hot encoded columns)
    ts_cols = [c for c in df.columns if 'ts_' in c]
    if ts_cols:
        ts_sums = df[ts_cols].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        ts_sums.plot(kind='bar', color='teal')
        plt.title("Count of Tracks by Time Signature")
        plt.savefig(f"{OUTPUT_DIR}/8_time_signature_counts.png")
        plt.close()
        print("Saved time signature analysis.")

    # 9. Hexbin plot for dense relationship visualization
    if 'valence' in df.columns and 'danceability' in df.columns:
        plt.figure(figsize=(10, 8))
        plt.hexbin(df['valence'], df['danceability'], gridsize=20, cmap='Blues')
        plt.colorbar(label='Count')
        plt.title("Hexbin: Valence (Mood) vs Danceability")
        plt.savefig(f"{OUTPUT_DIR}/9_mood_dance_hexbin.png")
        plt.close()
        print("Saved mood vs danceability hexbin plot.")

    # 10. Pairplot overview of selected key features
    subset_cols = ['popularity', 'energy', 'acousticness', 'instrumentalness']
    subset_cols = [c for c in subset_cols if c in df.columns]

    if subset_cols:
        subset = df[subset_cols].sample(min(500, len(df)))
        pp = sns.pairplot(subset, corner=True)
        pp.savefig(f"{OUTPUT_DIR}/10_key_features_pairplot.png")
        print("Saved key features pairplot.")

    print(f"EDA Complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
