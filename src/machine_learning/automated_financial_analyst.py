import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import schedule
import time
import requests
import subprocess
import sys
from pathlib import Path
from src.machine_learning.convert_text_to_pdf import convert_md_to_pdf

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

# MongoDB connection
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
database_name = os.getenv("DB_NAME")
uri = f"mongodb+srv://{username}:{password}@ol-cluster.3agvwhk.mongodb.net/{database_name}?retryWrites=true&w=majority"

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# File paths
FUNDAMENTAL_SCRIPT = "./src/machine_learning/fundamental_analysis.py"
FORECASTING_SCRIPT = "./src/machine_learning/stock_forecasting.py"
FUNDAMENTAL_OUTPUT = "nvidia_fundamental_features.csv"
FORECASTING_OUTPUT = "tuning_results.json"

# ==========================================
# 2. TELEGRAM FUNCTIONS
# ==========================================

def send_telegram_message(message, parse_mode='Markdown'):
    """Send text message to Telegram"""
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    
    # Clean message to avoid encoding issues
    if parse_mode == 'Markdown':
        # Escape special Markdown characters that might cause issues
        message = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    
    payload = {
        'chat_id': telegram_chat_id,
        'text': message,
        'parse_mode': parse_mode
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 400:
            # Try without parse_mode if Markdown fails
            print("Markdown failed, retrying without formatting...")
            payload['parse_mode'] = None
            response = requests.post(url, json=payload)
        
        response.raise_for_status()
        print("[OK] Message sent to Telegram successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to send message to Telegram: {e}")
        print(f"Message was: {message[:100]}...")
        return False

def send_telegram_document(file_path, caption=None):
    """Send document file to Telegram"""
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendDocument"
    
    try:
        with open(file_path, 'rb') as file:
            files = {'document': file}
            data = {'chat_id': telegram_chat_id}
            if caption:
                data['caption'] = caption
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print(f"✅ Document sent to Telegram: {file_path}")
            return True
    except Exception as e:
        print(f"❌ Failed to send document to Telegram: {e}")
        return False

# ==========================================
# 3. PIPELINE EXECUTION FUNCTIONS
# ==========================================

def run_python_script(script_path, script_name):
    """
    Execute a Python script and capture output
    
    Args:
        script_path: Path to the Python script
        script_name: Display name for logging
    
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    print(f"\n{'='*70}")
    print(f"🔄 Running: {script_name}")
    print(f"{'='*70}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True, result.stdout, ""
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        error_msg = f"⏱️ {script_name} timed out after 10 minutes"
        print(error_msg)
        return False, "", error_msg
    except Exception as e:
        error_msg = f"❌ Error running {script_name}: {str(e)}"
        print(error_msg)
        return False, "", error_msg

def verify_output_files():
    """Verify that required output files exist and are recent"""
    files_to_check = [
        (FUNDAMENTAL_OUTPUT, "Fundamental Analysis CSV"),
        (FORECASTING_OUTPUT, "Forecasting Results JSON")
    ]
    
    all_valid = True
    issues = []
    
    for file_path, file_name in files_to_check:
        if not Path(file_path).exists():
            issues.append(f"❌ Missing: {file_name} ({file_path})")
            all_valid = False
        else:
            # Check if file was modified in the last 24 hours
            file_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            if age_hours < 24:
                print(f"✅ {file_name} is up to date ({age_hours:.1f} hours old)")
            else:
                issues.append(f"⚠️ Warning: {file_name} is {age_hours:.1f} hours old")
    
    return all_valid, issues

# ==========================================
# 4. DATA LOADING FUNCTIONS
# ==========================================

def load_historical_prices(days=14):
    """Load last N days of stock prices from MongoDB"""
    print(f"Loading last {days} days of stock prices...")
    try:
        client = MongoClient(uri)
        db = client[database_name]
        collection = db["stock_prices"]
        
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
        with open(FORECASTING_OUTPUT, 'r') as f:
            data = json.load(f)
        return data['forecast_next_3_days']['ensemble']
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return []

def load_fundamental_features():
    """Load fundamental analysis from CSV"""
    print("Loading fundamental analysis...")
    try:
        df = pd.read_csv(FUNDAMENTAL_OUTPUT)
        df['day'] = pd.to_datetime(df['day'])
        df = df.sort_values('day').tail(14)
        return df
    except Exception as e:
        print(f"Error loading fundamentals: {e}")
        return pd.DataFrame()

# ==========================================
# 5. DATA FORMATTING FOR LLM
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
    
    # RAW DATA TABLE
    summary += "\nRAW DATA (Last 14 Days) - For Pattern Discovery:\n"
    summary += "-" * 60 + "\n"
    summary += f"{'Date':<12} {'Discord':<10} {'Fear':<6} {'Hype':<6} {'Sentiment':<11} {'Volume':<8} {'Accel':<8}\n"
    summary += "-" * 60 + "\n"
    
    for _, row in df.iterrows():
        date = pd.to_datetime(row['day']).strftime('%Y-%m-%d')
        summary += f"{date:<12} {row['sentiment_discord']:>8.4f}  {int(row['fear_index']):>4}  {int(row['product_hype_score']):>4}  {row['viral_sentiment']:>9.4f}  {int(row['post_volume']):>6}  {row['attention_acceleration']:>7.2f}\n"
    
    summary += "\n" + "=" * 60 + "\n"
    
    # Latest metrics
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
    
    # Statistics
    summary += f"\n14-DAY STATISTICAL SUMMARY:\n"
    summary += f"  - Sentiment Discord: min={df['sentiment_discord'].min():.4f}, max={df['sentiment_discord'].max():.4f}, mean={df['sentiment_discord'].mean():.4f}\n"
    summary += f"  - Fear Index: min={int(df['fear_index'].min())}, max={int(df['fear_index'].max())}, mean={df['fear_index'].mean():.2f}\n"
    summary += f"  - Product Hype: total_mentions={int(df['product_hype_score'].sum())}, days_with_mentions={int((df['product_hype_score'] > 0).sum())}\n"
    summary += f"  - Viral Sentiment: min={df['viral_sentiment'].min():.4f}, max={df['viral_sentiment'].max():.4f}, mean={df['viral_sentiment'].mean():.4f}\n"
    summary += f"  - Post Volume: min={int(df['post_volume'].min())}, max={int(df['post_volume'].max())}, mean={df['post_volume'].mean():.2f}\n"
    summary += f"  - Attention Acceleration: min={df['attention_acceleration'].min():.2f}, max={df['attention_acceleration'].max():.2f}\n"
    
    # Trends
    recent_7 = df.tail(7)
    previous_7 = df.iloc[-14:-7] if len(df) >= 14 else df.head(7)
    
    summary += f"\n7-DAY TREND ANALYSIS (Recent vs Previous Week):\n"
    summary += f"  - Avg Sentiment: {recent_7['viral_sentiment'].mean():.4f} vs {previous_7['viral_sentiment'].mean():.4f} "
    sentiment_change = recent_7['viral_sentiment'].mean() - previous_7['viral_sentiment'].mean()
    summary += f"({'↑ IMPROVING' if sentiment_change > 0.05 else '↓ DECLINING' if sentiment_change < -0.05 else '→ STABLE'})\n"
    summary += f"  - Avg Fear Level: {recent_7['fear_index'].mean():.2f} vs {previous_7['fear_index'].mean():.2f}\n"
    summary += f"  - Avg Post Volume: {recent_7['post_volume'].mean():.2f} vs {previous_7['post_volume'].mean():.2f}\n"
    summary += f"  - Total Product Mentions: {int(recent_7['product_hype_score'].sum())} (recent) vs {int(previous_7['product_hype_score'].sum())} (previous)\n"
    
    # Correlations
    if len(df) >= 7:
        summary += f"\nCORRELATION PATTERNS (for AI pattern discovery):\n"
        summary += f"  - Fear vs Sentiment: {df['fear_index'].corr(df['viral_sentiment']):.3f}\n"
        summary += f"  - Discord vs Volume: {df['sentiment_discord'].corr(df['post_volume']):.3f}\n"
        summary += f"  - Hype vs Sentiment: {df['product_hype_score'].corr(df['viral_sentiment']):.3f}\n"
        summary += f"  - Acceleration vs Sentiment: {df['attention_acceleration'].corr(df['viral_sentiment']):.3f}\n"
    
    return summary

# ==========================================
# 6. LLM ANALYST
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
    
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )
    
    context = f"""
{format_price_history(price_data)}

{format_predictions(predictions)}

{format_fundamental_analysis(fundamentals)}

ADDITIONAL CONTEXT:
- Current Date: {datetime.now().strftime('%Y-%m-%d')}
- Stock: NVIDIA Corporation (NVDA)
- Analysis Type: Quantitative Short-term Trading
"""
    
    messages = [
        SystemMessage(content=create_analyst_prompt()),
        HumanMessage(content=f"""Analyze the following NVIDIA stock data and provide a trading recommendation:

{context}

Provide your complete analysis now.""")
    ]
    
    print("\n🤖 Consulting AI Financial Analyst...\n")
    response = llm.invoke(messages)
    
    return response.content

# ==========================================
# 7. COMPLETE PIPELINE
# ==========================================

def run_complete_pipeline():
    """
    Run the complete analysis pipeline:
    1. Run fundamental_analysis.py
    2. Run stock_forecasting.py
    3. Generate LLM recommendation
    4. Send results to Telegram
    """
    
    print("\n" + "=" * 70)
    print(f"🚀 NVDA COMPLETE ANALYSIS PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    pipeline_start_time = datetime.now()
    
    # Send start notification
    start_msg = f"🚀 *NVDA Daily Pipeline Started*\n\n"
    start_msg += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    start_msg += f"Running:\n"
    start_msg += f"1️⃣ Fundamental Analysis\n"
    start_msg += f"2️⃣ Stock Forecasting\n"
    start_msg += f"3️⃣ AI Recommendation"
    send_telegram_message(start_msg, parse_mode='Markdown')
    
    # Step 1: Run Fundamental Analysis
    print("\n" + "="*70)
    print("STEP 1/3: FUNDAMENTAL ANALYSIS")
    print("="*70)
    
    fund_success, fund_output, fund_error = run_python_script(FUNDAMENTAL_SCRIPT, "Fundamental Analysis")
    
    if not fund_success:
        error_msg = f"❌ *Pipeline Failed - Step 1*\n\n"
        error_msg += f"Fundamental Analysis script failed\n\n"
        error_msg += f"Error: {fund_error[:500]}"
        send_telegram_message(error_msg, parse_mode='Markdown')
        return False
    
    # Step 2: Run Stock Forecasting
    print("\n" + "="*70)
    print("STEP 2/3: STOCK FORECASTING")
    print("="*70)
    
    forecast_success, forecast_output, forecast_error = run_python_script(FORECASTING_SCRIPT, "Stock Forecasting")
    
    if not forecast_success:
        error_msg = f"❌ *Pipeline Failed - Step 2*\n\n"
        error_msg += f"Stock Forecasting script failed\n\n"
        error_msg += f"Error: {forecast_error[:500]}"
        send_telegram_message(error_msg, parse_mode='Markdown')
        return False
    
    # Step 3: Verify output files
    print("\n" + "="*70)
    print("VERIFYING OUTPUT FILES")
    print("="*70)
    
    files_valid, issues = verify_output_files()
    
    if not files_valid:
        error_msg = f"❌ *Pipeline Failed - Missing Files*\n\n"
        error_msg += "\n".join(issues)
        send_telegram_message(error_msg, parse_mode='Markdown')
        return False
    
    if issues:
        for issue in issues:
            print(issue)
    
    # Step 4: Load data and run LLM analysis
    print("\n" + "="*70)
    print("STEP 3/3: AI RECOMMENDATION")
    print("="*70)
    
    try:
        price_data = load_historical_prices(14)
        predictions = load_predictions()
        fundamentals = load_fundamental_features()
        
        if price_data.empty or not predictions or fundamentals.empty:
            error_msg = "❌ *Pipeline Failed - Data Loading*\n\n"
            error_msg += "Could not load required data for analysis"
            send_telegram_message(error_msg, parse_mode='Markdown')
            return False
        
        # Run LLM analysis
        recommendation = analyze_with_groq(price_data, predictions, fundamentals)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_recommendation_{timestamp}.txt"
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"NVDA Trading Recommendation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(recommendation)
        
        print("=" * 70)
        print("AI ANALYST RECOMMENDATION")
        print("=" * 70)
        print(recommendation)
        print("=" * 70)
        
        # Calculate pipeline duration
        pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
        
        # Create summary for Telegram
        current_price = price_data['Close'].iloc[-1]
        price_change = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[0] - 1) * 100)
        
        summary = f"✅ *NVDA Daily Analysis Complete*\n\n"
        summary += f"⏱️ Pipeline duration: {pipeline_duration:.0f} seconds\n\n"
        summary += f"📊 *Market Data:*\n"
        summary += f"💰 Current Price: ${current_price:.2f}\n"
        summary += f"📈 14-Day Change: {price_change:+.2f}%\n"
        summary += f"🔮 3-Day Prediction: ${predictions[-1]:.2f}\n\n"
        summary += f"📝 Full analysis report attached below 👇"
        
        # Send summary
        send_telegram_message(summary, parse_mode='Markdown')
        
        # Send document
        # Convert to PDF for better formatting
        pdf_path = f"{os.path.splitext(os.path.basename(filename))[0]}_report.pdf"
        convert_md_to_pdf(filename, output_filename=pdf_path)
        send_telegram_document(pdf_path, caption="🤖 Complete AI Analysis Report")
        
        print(f"\n✅ Complete pipeline executed successfully in {pipeline_duration:.0f} seconds")
        print(f"✅ Report saved: {filename}")
        
        return True
        
    except Exception as e:
        error_msg = f"❌ *Analysis Failed*\n\nError: {str(e)}"
        send_telegram_message(error_msg, parse_mode='Markdown')
        print(f"❌ Error during analysis: {e}")
        return False

# ==========================================
# 8. SCHEDULER
# ==========================================

def start_scheduler(run_time="09:00"):
    """
    Start the scheduler to run complete pipeline daily
    
    Args:
        run_time: Time to run analysis (24h format, e.g., "09:00" for 9 AM)
    """
    
    print("=" * 70)
    print("NVDA AI ANALYST - COMPLETE AUTOMATION PIPELINE")
    print("=" * 70)
    print(f"⏰ Scheduled to run daily at {run_time}")
    print(f"📊 Pipeline includes:")
    print(f"   1. Fundamental Analysis (sentiment, social data)")
    print(f"   2. Stock Forecasting (ensemble model)")
    print(f"   3. AI Recommendation (Groq LLM)")
    print(f"📱 Telegram notifications enabled")
    print("=" * 70)
    
    # Validate configuration
    if not all([groq_api_key, telegram_bot_token, telegram_chat_id]):
        print("\n❌ ERROR: Missing configuration!")
        print("Please set in .env file:")
        print("  - GROQ_API_KEY")
        print("  - TELEGRAM_BOT_TOKEN")
        print("  - TELEGRAM_CHAT_ID")
        return
    
    # Verify scripts exist
    if not Path(FUNDAMENTAL_SCRIPT).exists():
        print(f"❌ ERROR: {FUNDAMENTAL_SCRIPT} not found!")
        return
    
    if not Path(FORECASTING_SCRIPT).exists():
        print(f"❌ ERROR: {FORECASTING_SCRIPT} not found!")
        return
    
    # Send startup notification
    startup_msg = f"🚀 *NVDA AI Pipeline Bot Started*\n\n"
    startup_msg += f"⏰ Daily execution at {run_time}\n"
    startup_msg += f"📊 Full pipeline automation\n"
    startup_msg += f"🤖 Powered by Groq LLM\n\n"
    startup_msg += f"Pipeline includes:\n"
    startup_msg += f"• Fundamental analysis\n"
    startup_msg += f"• Price forecasting\n"
    startup_msg += f"• AI recommendations\n\n"
    startup_msg += f"_Bot is now running..._"
    send_telegram_message(startup_msg, parse_mode='Markdown')
    
    # Schedule the job
    schedule.every().day.at(run_time).do(run_complete_pipeline)
    
    # Option to run immediately
    print("\n💡 Run complete pipeline now? (y/n): ", end='')
    choice = input().lower()
    if choice == 'y':
        run_complete_pipeline()
    
    # Keep running
    print("\n✅ Scheduler is running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\n⛔ Scheduler stopped by user")
        send_telegram_message("⛔ *NVDA AI Pipeline Bot Stopped*", parse_mode='Markdown')

# ==========================================
# 9. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("NVDA AI FINANCIAL ANALYST - Complete Automation Pipeline")
    print("=" * 70)
    print("\nOptions:")
    print("1. Run complete pipeline once now")
    print("2. Start automated daily scheduler")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        run_complete_pipeline()
    elif choice == "2":
        run_time = input("Enter daily run time (HH:MM, 24h format, e.g., 09:00): ").strip()
        if not run_time:
            run_time = "09:00"
        start_scheduler(run_time)
    else:
        print("Exiting...")
        sys.exit(0)