import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

# MongoDB connection
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
database_name = os.getenv("DB_NAME")
uri = f"mongodb+srv://{username}:{password}@ol-cluster.3agvwhk.mongodb.net/{database_name}?retryWrites=true&w=majority"

# Groq API key (add to your .env file: GROQ_API_KEY=your_key_here)
groq_api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# 2. DATA LOADING FUNCTIONS
# ==========================================

def load_historical_prices(days=14):
    """Load last N days of stock prices from MongoDB"""
    print(f"Loading last {days} days of stock prices...")
    try:
        client = MongoClient(uri)
        db = client[database_name]
        collection = db["stock_prices"]
        
        # Get last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        cursor = collection.find(
            {"timestamp": {"$gte": start_date}},
            {"timestamp": 1, "Close": 1, "Volume": 1, "High": 1, "Low": 1, "_id": 0}
        ).sort("timestamp", -1).limit(days)
        
        df = pd.DataFrame(list(cursor))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        print(f"Error loading prices: {e}")
        return pd.DataFrame()

def load_predictions():
    """Load ensemble predictions from tuning_results.json"""
    print("Loading predictions...")
    try:
        with open('tuning_results.json', 'r') as f:
            data = json.load(f)
        return data['forecast_next_3_days']['ensemble']
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return []

def load_fundamental_features():
    """Load fundamental analysis from CSV"""
    print("Loading fundamental analysis...")
    try:
        df = pd.read_csv('nvidia_fundamental_features.csv')
        # Get last 14 days
        df['day'] = pd.to_datetime(df['day'])
        df = df.sort_values('day').tail(14)
        return df
    except Exception as e:
        print(f"Error loading fundamentals: {e}")
        return pd.DataFrame()

# ==========================================
# 3. DATA FORMATTING FOR LLM
# ==========================================

def format_price_history(df):
    """Format price data for LLM consumption"""
    if df.empty:
        return "No price data available."
    
    summary = "HISTORICAL PRICE DATA (Last 14 Days):\n"
    summary += "=" * 60 + "\n"
    
    for _, row in df.iterrows():
        date = row['timestamp'].strftime('%Y-%m-%d')
        summary += f"{date}: Close=${row['Close']:.2f}, High=${row['High']:.2f}, Low=${row['Low']:.2f}, Volume={row['Volume']:,.0f}\n"
    
    # Add statistics
    summary += "\n" + "=" * 60 + "\n"
    summary += f"Period Statistics:\n"
    summary += f"  - Average Close: ${df['Close'].mean():.2f}\n"
    summary += f"  - High: ${df['High'].max():.2f}\n"
    summary += f"  - Low: ${df['Low'].min():.2f}\n"
    summary += f"  - Price Change: {((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100):.2f}%\n"
    summary += f"  - Volatility (Std Dev): ${df['Close'].std():.2f}\n"
    
    return summary

def format_predictions(predictions):
    """Format prediction data"""
    if not predictions:
        return "No predictions available."
    
    summary = "\nENSEMBLE MODEL PREDICTIONS (Next 3 Days):\n"
    summary += "=" * 60 + "\n"
    
    today = datetime.now()
    for i, price in enumerate(predictions):
        future_date = (today + timedelta(days=i+1)).strftime('%Y-%m-%d')
        summary += f"Day {i+1} ({future_date}): ${price:.2f}\n"
    
    summary += f"\nPredicted 3-Day Return: {((predictions[-1] / predictions[0] - 1) * 100):.2f}%\n"
    
    return summary

def format_fundamental_analysis(df):
    """Format fundamental features with RAW VALUES and interpretations"""
    if df.empty:
        return "No fundamental data available."
    
    summary = "\nFUNDAMENTAL ANALYSIS (Social & Sentiment Metrics):\n"
    summary += "=" * 60 + "\n"
    
    # RAW DATA TABLE - Last 14 Days
    summary += "\nRAW DATA (Last 14 Days) - For Pattern Discovery:\n"
    summary += "-" * 60 + "\n"
    summary += f"{'Date':<12} {'Discord':<10} {'Fear':<6} {'Hype':<6} {'Sentiment':<11} {'Volume':<8} {'Accel':<8}\n"
    summary += "-" * 60 + "\n"
    
    for _, row in df.iterrows():
        date = pd.to_datetime(row['day']).strftime('%Y-%m-%d')
        summary += f"{date:<12} {row['sentiment_discord']:>8.4f}  {int(row['fear_index']):>4}  {int(row['product_hype_score']):>4}  {row['viral_sentiment']:>9.4f}  {int(row['post_volume']):>6}  {row['attention_acceleration']:>7.2f}\n"
    
    summary += "\n" + "=" * 60 + "\n"
    
    # Latest metrics with interpretation
    latest = df.iloc[-1]
    summary += f"\nLATEST METRICS ({pd.to_datetime(latest['day']).strftime('%Y-%m-%d')}):\n"
    summary += f"  - Sentiment Discord: {latest['sentiment_discord']:.4f} "
    summary += f"({'HIGH UNCERTAINTY' if latest['sentiment_discord'] > 0.4 else 'CONSENSUS'})\n"
    
    summary += f"  - Fear Index: {int(latest['fear_index'])} "
    summary += f"({'PANIC MODE' if latest['fear_index'] > 3 else 'CALM'})\n"
    
    summary += f"  - Product Hype Score: {int(latest['product_hype_score'])} "
    summary += f"({'Strong Fundamental Focus' if latest['product_hype_score'] > 0 else 'No product mentions'})\n"
    
    summary += f"  - Viral Sentiment: {latest['viral_sentiment']:.4f} "
    if latest['viral_sentiment'] > 0.2:
        summary += "(BULLISH)\n"
    elif latest['viral_sentiment'] < -0.2:
        summary += "(BEARISH)\n"
    else:
        summary += "(NEUTRAL)\n"
    
    summary += f"  - Post Volume: {int(latest['post_volume'])}\n"
    summary += f"  - Attention Acceleration: {latest['attention_acceleration']:.2f} "
    summary += f"({'VIRAL EVENT!' if latest['attention_acceleration'] > 1.0 else 'Normal'})\n"
    
    # Statistical Summary
    summary += f"\n14-DAY STATISTICAL SUMMARY:\n"
    summary += f"  - Sentiment Discord: min={df['sentiment_discord'].min():.4f}, max={df['sentiment_discord'].max():.4f}, mean={df['sentiment_discord'].mean():.4f}\n"
    summary += f"  - Fear Index: min={int(df['fear_index'].min())}, max={int(df['fear_index'].max())}, mean={df['fear_index'].mean():.2f}\n"
    summary += f"  - Product Hype: total_mentions={int(df['product_hype_score'].sum())}, days_with_mentions={int((df['product_hype_score'] > 0).sum())}\n"
    summary += f"  - Viral Sentiment: min={df['viral_sentiment'].min():.4f}, max={df['viral_sentiment'].max():.4f}, mean={df['viral_sentiment'].mean():.4f}\n"
    summary += f"  - Post Volume: min={int(df['post_volume'].min())}, max={int(df['post_volume'].max())}, mean={df['post_volume'].mean():.2f}\n"
    summary += f"  - Attention Acceleration: min={df['attention_acceleration'].min():.2f}, max={df['attention_acceleration'].max():.2f}\n"
    
    # Trend analysis (last 7 days vs previous 7 days)
    recent_7 = df.tail(7)
    previous_7 = df.iloc[-14:-7] if len(df) >= 14 else df.head(7)
    
    summary += f"\n7-DAY TREND ANALYSIS (Recent vs Previous Week):\n"
    summary += f"  - Avg Sentiment: {recent_7['viral_sentiment'].mean():.4f} vs {previous_7['viral_sentiment'].mean():.4f} "
    sentiment_change = recent_7['viral_sentiment'].mean() - previous_7['viral_sentiment'].mean()
    summary += f"({'↑ IMPROVING' if sentiment_change > 0.05 else '↓ DECLINING' if sentiment_change < -0.05 else '→ STABLE'})\n"
    
    summary += f"  - Avg Fear Level: {recent_7['fear_index'].mean():.2f} vs {previous_7['fear_index'].mean():.2f}\n"
    summary += f"  - Avg Post Volume: {recent_7['post_volume'].mean():.2f} vs {previous_7['post_volume'].mean():.2f}\n"
    summary += f"  - Total Product Mentions: {int(recent_7['product_hype_score'].sum())} (recent) vs {int(previous_7['product_hype_score'].sum())} (previous)\n"
    
    # Correlation insights (if enough data)
    if len(df) >= 7:
        summary += f"\nCORRELATION PATTERNS (for AI pattern discovery):\n"
        summary += f"  - Fear vs Sentiment: {df['fear_index'].corr(df['viral_sentiment']):.3f}\n"
        summary += f"  - Discord vs Volume: {df['sentiment_discord'].corr(df['post_volume']):.3f}\n"
        summary += f"  - Hype vs Sentiment: {df['product_hype_score'].corr(df['viral_sentiment']):.3f}\n"
        summary += f"  - Acceleration vs Sentiment: {df['attention_acceleration'].corr(df['viral_sentiment']):.3f}\n"
    
    return summary

# ==========================================
# 4. LLM ANALYST
# ==========================================

def create_analyst_prompt():
    """Create the system prompt for the financial analyst"""
    return """You are a **Senior Quantitative Financial Analyst** specializing in equity trading and advanced pattern recognition. 

Your role is to analyze NVIDIA (NVDA) stock data and provide **actionable trading recommendations** with precise justifications.

**Your Analysis Framework:**

1. **Technical Analysis** (Price Action)
   - Identify trends (uptrend, downtrend, consolidation)
   - Analyze support/resistance levels
   - Evaluate momentum and volatility

2. **Predictive Model Confidence**
   - Assess the ensemble model's forecasted direction
   - Consider prediction magnitude and consistency

3. **Fundamental Sentiment Analysis** (You have RAW DATA for pattern discovery)
   - sentiment_discord: High values (>0.4) = uncertainty/volatility ahead
   - fear_index: High values (>3) = panic selling (contrarian buy signal if oversold)
   - product_hype_score: >0 = fundamental catalysts present
   - viral_sentiment: >0.2 = bullish, <-0.2 = bearish
   - attention_acceleration: >1.0 = breaking news/viral event
   
   **CRITICAL: You have access to raw time-series data. Look for:**
   - Hidden sequential patterns (e.g., fear spikes followed by rebounds)
   - Divergences (sentiment rising while price falling = bullish divergence)
   - Correlation anomalies (unusual relationships between metrics)
   - Regime changes (sudden shifts in multiple indicators)
   - Leading indicators (which metrics predict price movements)

4. **Risk Assessment**
   - Market regime (trending vs. mean-reverting)
   - Position sizing based on confidence level

5. **Pattern Discovery** (IMPORTANT)
   - Analyze the raw fundamental data table for non-obvious patterns
   - Identify early warning signals from metric combinations
   - Detect regime shifts or anomalies in the time series
   - Consider lagged correlations (today's sentiment → tomorrow's price)

**Output Format (STRICT):**

RECOMMENDATION: [BUY/SELL/HOLD]
POSITION SIZE: [Conservative/Moderate/Aggressive] - [% of portfolio]
CONFIDENCE: [Low/Medium/High]

PRICE TARGETS:
- Entry Price: $XXX.XX
- Target Price (3-day): $XXX.XX
- Stop Loss: $XXX.XX

JUSTIFICATION:
[Provide 3-5 bullet points explaining your decision based on the data]

HIDDEN PATTERNS DISCOVERED:
[List any non-obvious patterns or correlations you found in the raw data]

RISK FACTORS:
[List 2-3 key risks to this trade]

TIME HORIZON: [Short-term (1-3 days) / Medium-term (1-2 weeks)]

---
Be direct, quantitative, and actionable. Use the raw data to discover insights beyond simple thresholds."""

def analyze_with_groq(price_data, predictions, fundamentals):
    """Send data to Groq LLM and get trading recommendation"""
    
    # Initialize Groq LLM
    llm = ChatGroq(
        temperature=0.1,  # Low temperature for consistent, factual analysis
        model_name="llama-3.3-70b-versatile",  # Fast and capable model
        groq_api_key=groq_api_key
    )
    
    # Prepare the analysis context
    context = f"""
{format_price_history(price_data)}

{format_predictions(predictions)}

{format_fundamental_analysis(fundamentals)}

ADDITIONAL CONTEXT:
- Current Date: {datetime.now().strftime('%Y-%m-%d')}
- Stock: NVIDIA Corporation (NVDA)
- Analysis Type: Quantitative Short-term Trading
"""
    
    # Create messages
    messages = [
        SystemMessage(content=create_analyst_prompt()),
        HumanMessage(content=f"""Analyze the following NVIDIA stock data and provide a trading recommendation:

{context}

Provide your complete analysis now.""")
    ]
    
    # Get response
    print("\n🤖 Consulting AI Financial Analyst...\n")
    response = llm.invoke(messages)
    
    return response.content

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    print("=" * 70)
    print("NVDA AI FINANCIAL ANALYST - Powered by Groq LLM")
    print("=" * 70)
    
    # Load all data
    price_data = load_historical_prices(14)
    predictions = load_predictions()
    fundamentals = load_fundamental_features()
    
    # Validate data
    if price_data.empty or not predictions or fundamentals.empty:
        print("❌ Error: Missing required data files")
        print("Ensure the following files exist:")
        print("  - MongoDB connection for stock prices")
        print("  - tuning_results.json")
        print("  - nvidia_fundamental_features.csv")
        return
    
    # Run analysis
    try:
        recommendation = analyze_with_groq(price_data, predictions, fundamentals)
        
        # Display results
        print("=" * 70)
        print("AI ANALYST RECOMMENDATION")
        print("=" * 70)
        print(recommendation)
        print("=" * 70)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"trading_recommendation_{timestamp}.txt", "w") as f:
            f.write(f"NVDA Trading Recommendation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(recommendation)
        
        print(f"\n✅ Recommendation saved to trading_recommendation_{timestamp}.txt")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure you have set GROQ_API_KEY in your .env file")

if __name__ == "__main__":
    main()