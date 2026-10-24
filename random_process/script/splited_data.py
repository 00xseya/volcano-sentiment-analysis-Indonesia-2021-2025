import pandas as pd
from pathlib import Path

SOURCE_DATA = Path("../data/ner_distribution/data_text.csv")
OUTPUT_DIR = Path("random/output")
OUTPUT_FILE = OUTPUT_DIR / "splited_data.csv"
OUTPUT_FILE.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE_DATA)
print(f"Total of rows: {len{df};,}\n")

print("Year distribution in original data:")
print(df["video_year"].value_counts().sort_index())
print()

# Stratified 30% 
sample_30 = (
    df.groupby("video_year", group_keys=False)
      .apply(lambda g: g.sample(frac=0.30, random_state=42))
      .reset_index(drop=True)
)

# Remaining 70%
remaining_70 = df.drop(sample_30.index).reset_index(drop=True)

print(f"30% sample:    {len(sample_30):,} rows")
print(f"70% remaining: {len(remaining_70):,} rows\n")

print("Year distribution in 30% sample:")
print(sample_30["video_year"].value_counts().sort_index())

# Save
sample_30.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
remaining_70.to_csv(OUTPUT_FILE / "comments_remaining_70pct_stratified.csv", index=False, encoding="utf-8")

print(f"\n Saved: {OUTPUT_FILE / 'comments_sample_30pct_stratified.csv'}")
print(f" Saved: {OUTPUT_FILE / 'comments_remaining_70pct_stratified.csv'}")