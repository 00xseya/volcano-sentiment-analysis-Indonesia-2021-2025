import pandas as pd
from pathlib import Path

SOURCE_FILE = Path("data/cleansing/youtube_comments_cleansed.csv")
OUTPUT_DIR = Path("data/ner_distribution")
OUTPUT_FILE = OUTPUT_DIR / "data_text.csv"

df = pd.read_csv(SOURCE_FILE)

data_df = df[['video_title', 'video_year', 'text_cleansed']]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully extracted {len(data_df)} rows to {OUTPUT_FILE}")