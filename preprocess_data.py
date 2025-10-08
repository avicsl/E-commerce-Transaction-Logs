import pandas as pd
import re

SUSPICIOUS_KEYWORDS = [
    'breaking', 'shocking', 'secret', 'banned', 'miracle', 'urgent',
    'exposed', 'revealed', 'hidden', 'cure', 'conspiracy', 'government',
    'cover-up', 'they don\'t want you', 'must see', 'click now',
    'you won\'t believe', 'doctors hate', 'wake up', 'share before'
]

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_suspicious_keywords(text):
    if pd.isna(text) or text == "":
        return 0
    text_lower = text.lower()
    return sum(keyword in text_lower for keyword in SUSPICIOUS_KEYWORDS)

def clean_and_engineer(df):
    # Fill missing values, but never drop any rows
    df['source_link'] = df['source_link'].fillna('UNKNOWN')
    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # If timestamp invalid, fill with median date (preserves all 500 rows)
    median_time = df['timestamp'].dropna().median()
    df['timestamp'] = df['timestamp'].fillna(median_time)
    
    # Normalize post content
    df['post_content_normalized'] = df['post_content'].apply(normalize_text)

    # Feature engineering
    df['word_count'] = df['post_content_normalized'].apply(lambda x: len(x.split()))
    df['contains_link'] = df['source_link'].apply(lambda x: x != "UNKNOWN")
    df['emotion_intensity'] = df['sentiment_score'].abs()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['engagement_rate'] = df['num_shares'] + df['num_reactions']
    df['suspicious_keyword_count'] = df['post_content_normalized'].apply(count_suspicious_keywords)
    df['day_of_week'] = df['timestamp'].dt.day_name()

    return df

def main():
    INPUT_FILE = 'final_project_raw_data.csv'
    OUTPUT_FILE = 'final_project_cleaned_data.csv'

    df = pd.read_csv(INPUT_FILE)
    cleaned = clean_and_engineer(df)
    cleaned.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Cleaned dataset saved to {OUTPUT_FILE}")
    print(f"📊 Total rows retained: {len(cleaned)} (no data loss)")

if __name__ == "__main__":
    main()

