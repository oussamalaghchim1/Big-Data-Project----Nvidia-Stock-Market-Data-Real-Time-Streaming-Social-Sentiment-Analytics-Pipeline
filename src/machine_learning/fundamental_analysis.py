import pandas as pd
import numpy as np
from pymongo import MongoClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone
import re
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Download VADER lexicon if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize Analyzer
sia = SentimentIntensityAnalyzer()

# --- CUSTOM FINANCIAL DICTIONARY ---
# We teach the model slang that standard NLP misses
financial_lexicon = {
    'moon': 2.0,       'rocket': 2.0,     'lambo': 2.0,
    'calls': 1.0,      'long': 1.0,       'bull': 1.5,
    'breakout': 1.5,   'green': 1.0,      'print': 1.0,
    'puts': -1.0,      'short': -1.5,     'bear': -1.5,
    'crash': -2.5,     'dump': -2.0,      'drill': -2.0,
    'bagholder': -1.5, 'ripping': 1.5,    'tanking': -1.5,
    'rug pull': -2.0,  'gem': 1.5,        'fud': -1.0
}
sia.lexicon.update(financial_lexicon)

# Database Connection
# Load environment variables
load_dotenv()
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
database_name = os.getenv("DB_NAME")
uri = f"mongodb+srv://{username}:{password}@ol-cluster.3agvwhk.mongodb.net/{database_name}?retryWrites=true&w=majority"
print("Connecting to MongoDB Atlas...")
try:
    client = MongoClient(uri)
    db = client[database_name]
    collection = db["stock_prices"]
    print(f'Documents found: {collection.count_documents({})}')
except Exception as e:
    print(f"❌ Connection Error: {e}")
    exit()



# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_vader_score(text):
    """Returns the compound score (-1 to 1)."""
    if not isinstance(text, str): return 0
    return sia.polarity_scores(text)['compound']

def count_keywords(text, keywords):
    """Counts occurrences of specific keywords."""
    if not isinstance(text, str): return 0
    text_lower = text.lower()
    return sum(1 for word in keywords if word in text_lower)

# ==========================================
# 3. DATA LOADING & FILTERING
# ==========================================

def load_filtered_data():
    # SIMULATION DATE: Adjusted for your 2026 dataset
    # If running for real-time today, use: today = datetime.now()
    today = datetime.now(timezone.utc) 
    start_date = today - timedelta(days=30)

    print(f"--- Loading data from {start_date.date()} to {today.date()} ---")

    # --- A. REDDIT FILTERING ---
    # Filter: Last 30 days + Keywords + Not Deleted
    reddit_query = {
        "timestamp": {"$gte": start_date},
        "$and": [
            { "texte": {"$ne": "[removed]"} },
            { "texte": {"$ne": "[deleted]"} }
        ],
        "$or": [
            {"titre": {"$regex": "NVDA|Nvidia|Jensen|Blackwell|RTX|GPU", "$options": "i"}},
            {"texte": {"$regex": "NVDA|Nvidia|Jensen|Blackwell|RTX|GPU", "$options": "i"}}
        ]
    }
    
    df_reddit = pd.DataFrame(list(db.nvidia_reddit_posts.find(reddit_query)))
    if not df_reddit.empty:
        df_reddit['date'] = pd.to_datetime(df_reddit['timestamp'])
        df_reddit['text_full'] = df_reddit['titre'] + " " + df_reddit['texte']
        df_reddit['likes'] = df_reddit['score'].fillna(0)
        df_reddit['source'] = 'reddit'

    # --- B. STOCKTWITS FILTERING ---
    # Filter: Last 30 days (Data is usually already ticker-specific)
    st_query = {"timestamp": {"$gte": start_date}}
    df_st = pd.DataFrame(list(db.nvidia_stockwits_posts.find(st_query)))
    if not df_st.empty:
        df_st['date'] = pd.to_datetime(df_st['timestamp'])
        df_st['text_full'] = df_st['text']
        df_st['likes'] = df_st['likes'].fillna(0)
        df_st['source'] = 'stocktwits'
        # Map explicit labels to scores
        sentiment_map = {'Bullish': 1.0, 'Bearish': -1.0}
        df_st['manual_sentiment'] = df_st['sentiment'].map(sentiment_map)

    # --- C. YAHOO NEWS FILTERING ---
    # Note: Using python-side filtering for dates if stored as strings
    df_yahoo = pd.DataFrame(list(db.nvidia_yahoo_news.find({
        "$or": [
            {"title": {"$regex": "NVDA|Nvidia", "$options": "i"}},
            {"summary": {"$regex": "NVDA|Nvidia", "$options": "i"}}
        ]
    })))
    
    if not df_yahoo.empty:
        df_yahoo['date'] = pd.to_datetime(df_yahoo['pubdate'])
        # Apply date filter in Pandas just to be safe with formats
        df_yahoo = df_yahoo[df_yahoo['date'] >= start_date]
        df_yahoo['text_full'] = df_yahoo['title'] + " " + df_yahoo['summary']
        df_yahoo['likes'] = 5 # Higher weight for news authority
        df_yahoo['source'] = 'news'

    # ... (le début de la fonction reste identique)

    # --- MERGE ---
    cols = ['date', 'text_full', 'likes', 'source']
    if not df_st.empty and 'manual_sentiment' in df_st.columns:
        cols.append('manual_sentiment')

    frames = [d for d in [df_reddit, df_st, df_yahoo] if not d.empty]
    
    if not frames:
        return pd.DataFrame()
        
    df_all = pd.concat([d[cols] for d in frames], ignore_index=True)
    
    # --- CORRECTION CRITIQUE ICI ---
    # On force à nouveau le format datetime sur la colonne finale pour être sûr à 100%
    # errors='coerce' va transformer les dates invalides en NaT (Not a Time) au lieu de planter
    df_all['date'] = pd.to_datetime(df_all['date'], utc=True, errors='coerce')
    
    # On supprime les lignes où la date n'a pas pu être convertie (sécurité)
    df_all = df_all.dropna(subset=['date'])

    # Maintenant .dt fonctionnera sans erreur
    df_all['day'] = df_all['date'].dt.date
    
    return df_all

# ==========================================
# 4. FEATURE ENGINEERING PIPELINE
# ==========================================

def generate_fundamental_features():
    df = load_filtered_data()
    
    if df.empty:
        print("No relevant data found for the period.")
        return None

    print(f"Processing {len(df)} combined documents...")

    # --- STEP 1: Row-Level Calculations ---
    
    # 1.1 Sentiment Score (VADER Enhanced)
    df['sentiment_score'] = df['text_full'].apply(get_vader_score)
    
    # Override with manual sentiment if available (StockTwits)
    if 'manual_sentiment' in df.columns:
        df['sentiment_score'] = df['manual_sentiment'].fillna(df['sentiment_score'])

    # 1.2 Fear Keywords Detection
    fear_keywords = ["crash", "bubble", "panic", "dump", "puts", "sell", "bear", "plunge", "recession"]
    df['fear_count'] = df['text_full'].apply(lambda x: count_keywords(x, fear_keywords))

    # 1.3 Entity/Product Recognition
    product_keywords = ["blackwell", "rtx5090", "h100", "h200", "rubin", "b100", "b200"]
    df['product_mention'] = df['text_full'].apply(lambda x: count_keywords(x, product_keywords))

    # --- STEP 2: Daily Aggregation ---
    
    daily_groups = df.groupby('day')
    features = pd.DataFrame()

    # Metric 1: Discord (Sentiment Standard Deviation)
    features['sentiment_discord'] = daily_groups['sentiment_score'].std().fillna(0)

    # Metric 2: Fear Index (Sum of fear words)
    features['fear_index'] = daily_groups['fear_count'].sum()

    # Metric 3: Product Hype (Sum of product mentions)
    features['product_hype_score'] = daily_groups['product_mention'].sum()

    # Metric 4: Viral Score (Weighted Sentiment)
    def calculate_viral_score(group):
        total_likes = group['likes'].sum()
        if total_likes == 0:
            return group['sentiment_score'].mean()
        # Weighted Average: (Sentiment * Likes) / Total Likes
        return (group['sentiment_score'] * group['likes']).sum() / total_likes

    features['viral_sentiment'] = daily_groups.apply(calculate_viral_score)

    # Metric 5: Attention Acceleration (Social Momentum)
    features['post_volume'] = daily_groups.size()
    
    # Ensure chronological order for rolling window
    features = features.sort_index()
    
    # Calculate 7-day Moving Average of Volume
    features['vol_ma_7d'] = features['post_volume'].rolling(window=7, min_periods=1).mean()
    
    # Momentum Formula: (Today - MA7) / MA7
    features['attention_acceleration'] = (features['post_volume'] - features['vol_ma_7d']) / features['vol_ma_7d']
    features['attention_acceleration'] = features['attention_acceleration'].fillna(0)

    # Cleanup intermediate columns
    features = features.drop(columns=['vol_ma_7d'])

    return features

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    final_features = generate_fundamental_features()
    
    if final_features is not None:
        print("\n--- Final Features Head (First 5 days) ---")
        print(final_features.head())
        print("\n--- Final Features Tail (Last 5 days) ---")
        print(final_features.tail())
        
        # Save to CSV for merging with technical data later
        final_features.to_csv("nvidia_fundamental_features.csv")
        print("\nSaved to nvidia_fundamental_features.csv")