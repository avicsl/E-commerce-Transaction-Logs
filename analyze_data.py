import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import spaCy (optional)
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False

def load_cleaned_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_entities(df):
    if not SPACY_AVAILABLE:
        df['entities'] = "N/A"
        return df
    
    entities_list = []
    for text in df['post_content']:
        if pd.isna(text):
            entities_list.append("NONE")
            continue
        doc = nlp(str(text)[:500])
        entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        entities_list.append(", ".join(entities) if entities else "NONE")
    
    df['entities'] = entities_list
    return df

def detect_anomalies(df):
    features = ['engagement_rate', 'emotion_intensity', 'suspicious_keyword_count', 
                'sentiment_score', 'word_count']
    
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        contamination=0.15,
        random_state=42,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    df['is_anomaly'] = predictions == -1
    
    return df

def flag_misinformation(df):
    df['is_misinformation'] = (
        (df['is_anomaly'] == True) |
        (df['suspicious_keyword_count'] >= 3) |
        ((df['engagement_rate'] > df['engagement_rate'].quantile(0.90)) & 
         (df['sentiment_score'] < 0))
    )
    return df

def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Facebook Post Misinformation Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Chart 1: Legitimate vs Misinformation Posts
    ax1 = axes[0, 0]
    misinfo_counts = df['is_misinformation'].value_counts()
    categories = ['Legitimate', 'Misinformation']
    values = [misinfo_counts.get(False, 0), misinfo_counts.get(True, 0)]
    colors = ['#5DADE2', '#1A5490']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Posts', fontweight='bold')
    ax1.set_title('Post Classification Results', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Sentiment Distribution
    ax2 = axes[0, 1]
    def categorize_sentiment(score):
        if score > 0.2:
            return 'Positive'
        elif score < -0.2:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    sentiment_counts = df['sentiment_category'].value_counts()
    colors_pie = ['#85C1E9', '#5DADE2', '#3498DB']
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax2.set_title('Sentiment Distribution', fontweight='bold')
    
    # Chart 3: Misinformation Over Time
    ax3 = axes[1, 0]
    df['date'] = df['timestamp'].dt.date
    daily_misinfo = df[df['is_misinformation'] == True].groupby('date').size()
    daily_total = df.groupby('date').size()
    
    ax3.plot(daily_misinfo.index, daily_misinfo.values, marker='o', 
             color='#1A5490', linewidth=2, markersize=6, label='Misinformation')
    ax3.plot(daily_total.index, daily_total.values, marker='s', 
             color='#5DADE2', linewidth=2, markersize=6, alpha=0.7, label='All Posts')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_ylabel('Number of Posts', fontweight='bold')
    ax3.set_title('Misinformation Posts Over Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Chart 4: Top Suspicious Domains
    ax4 = axes[1, 1]
    misinfo_posts = df[df['is_misinformation'] == True]
    suspicious_links = misinfo_posts[misinfo_posts['source_link'] != 'UNKNOWN']
    
    if len(suspicious_links) > 0:
        domain_counts = suspicious_links['source_link'].value_counts().head(8)
        ax4.barh(range(len(domain_counts)), domain_counts.values, color='#2E86C1', alpha=0.7)
        ax4.set_yticks(range(len(domain_counts)))
        ax4.set_yticklabels(domain_counts.index, fontsize=9)
        ax4.set_xlabel('Number of Posts', fontweight='bold')
        ax4.set_title('Top Suspicious Domains', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(domain_counts.values):
            ax4.text(v, i, f' {v}', va='center', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No suspicious domains with links', 
                ha='center', va='center', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    
    plt.tight_layout()
    output_file = 'final_project_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return output_file

def save_results(df):
    output_cols = ['post_id', 'user_id', 'timestamp', 'post_content', 
                   'num_shares', 'num_reactions', 'source_link', 
                   'sentiment_score', 'suspicious_keyword_count', 
                   'engagement_rate', 'is_anomaly', 'is_misinformation', 'entities']
    
    df_output = df[output_cols].copy()
    output_file = 'final_project_misinformation.csv'
    df_output.to_csv(output_file, index=False)
    return output_file

def main():
    INPUT_FILE = 'final_project_cleaned_data.csv'
    
    df = load_cleaned_data(INPUT_FILE)
    df = extract_entities(df)
    df = detect_anomalies(df)
    df = flag_misinformation(df)
    
    chart_file = create_visualizations(df)
    results_file = save_results(df)
    
    print(f"✅ Analysis complete!")
    print(f"✅ Misinformation results saved to {results_file}")
    print(f"✅ Visualizations saved to {chart_file}")

if __name__ == "__main__":

    main()
