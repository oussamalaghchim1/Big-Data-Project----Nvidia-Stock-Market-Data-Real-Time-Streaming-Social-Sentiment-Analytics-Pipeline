# Ajouté au début du script
import sys


if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from itertools import product
import numpy as np

# Darts imports
from darts import TimeSeries
from darts.models import Prophet, XGBModel
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mape, rmse
from darts.dataprocessing.transformers import Scaler

# Load environment variables
load_dotenv()
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
database_name = os.getenv("DB_NAME")
uri = f"mongodb+srv://{username}:{password}@ol-cluster.3agvwhk.mongodb.net/{database_name}?retryWrites=true&w=majority"

# ==========================================
# 1. CONNECT & FETCH DATA
# ==========================================
print("Connecting to MongoDB Atlas...")
try:
    client = MongoClient(uri)
    db = client[database_name]
    collection = db["stock_prices"]
    print(f'Documents found: {collection.count_documents({})}')
except Exception as e:
    print(f"Connection Error: {e}")
    exit()

print("Fetching data (Close + Volume)...")
cursor = collection.find({}, {"timestamp": 1, "Close": 1, "Volume": 1, "_id": 0}).sort("timestamp", 1)
df = pd.DataFrame(list(cursor))

if df.empty:
    print("Error: DataFrame is empty.")
    exit()

# ==========================================
# 2. PREPROCESS DATA
# ==========================================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
df['timestamp'] = df['timestamp'].dt.normalize()
df = df.drop_duplicates(subset='timestamp', keep='last')

# Create TWO TimeSeries: One for Price (Target), One for Volume (Covariate)
price_series = TimeSeries.from_dataframe(df, time_col='timestamp', value_cols='Close', fill_missing_dates=True, freq='D')
vol_series = TimeSeries.from_dataframe(df, time_col='timestamp', value_cols='Volume', fill_missing_dates=True, freq='D')

# Fill missing values for both series
price_series = fill_missing_values(price_series)
vol_series = fill_missing_values(vol_series)

# ==========================================
# 3. SCALING (CRITICAL STEP)
# ==========================================
print("Scaling data...")
price_scaler = Scaler()
vol_scaler = Scaler()

price_series_scaled = price_scaler.fit_transform(price_series)
vol_series_scaled = vol_scaler.fit_transform(vol_series)

# ==========================================
# 4. SPLIT DATA FOR VALIDATION
# ==========================================
# Use last 14 days for validation (better for tuning)
val_days = 14
train_scaled, val_scaled = price_series_scaled.split_before(len(price_series) - val_days)
train_vol_scaled, val_vol_scaled = vol_series_scaled.split_before(len(price_series) - val_days)
train_orig, val_orig = price_series.split_before(len(price_series) - val_days)

print(f"\nTraining on {len(train_orig)} days, Validating on {len(val_orig)} days...")

# ==========================================
# 5. HYPERPARAMETER TUNING - PROPHET
# ==========================================
print("\n" + "="*60)
print("PHASE 1: HYPERPARAMETER TUNING - PROPHET")
print("="*60)

# Prophet hyperparameter grid
prophet_params = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_range': [0.8, 0.9, 0.95]
}

best_prophet_mape = float('inf')
best_prophet_params = None
best_prophet_pred = None

print(f"Testing {len(prophet_params['changepoint_prior_scale']) * len(prophet_params['seasonality_prior_scale']) * len(prophet_params['seasonality_mode']) * len(prophet_params['changepoint_range'])} combinations...")

iteration = 0
for cp_scale, seas_scale, seas_mode, cp_range in product(
    prophet_params['changepoint_prior_scale'],
    prophet_params['seasonality_prior_scale'],
    prophet_params['seasonality_mode'],
    prophet_params['changepoint_range']
):
    iteration += 1
    try:
        model = Prophet(
            changepoint_prior_scale=cp_scale,
            seasonality_prior_scale=seas_scale,
            seasonality_mode=seas_mode,
            changepoint_range=cp_range
        )
        model.fit(train_orig)
        pred = model.predict(len(val_orig))
        current_mape = mape(val_orig, pred)
        
        if current_mape < best_prophet_mape:
            best_prophet_mape = current_mape
            best_prophet_params = {
                'changepoint_prior_scale': cp_scale,
                'seasonality_prior_scale': seas_scale,
                'seasonality_mode': seas_mode,
                'changepoint_range': cp_range
            }
            best_prophet_pred = pred
            print(f"✓ Iteration {iteration}: New best MAPE = {current_mape:.4f}% | Params: {best_prophet_params}")
    except Exception as e:
        print(f"✗ Iteration {iteration} failed: {e}")
        continue

p_mape = mape(val_orig, best_prophet_pred)
p_rmse = rmse(val_orig, best_prophet_pred)

print(f"\nBest Prophet MAPE: {p_mape:.2f}%")
print(f"Best Prophet RMSE: ${p_rmse:.2f}")
print(f"Best Parameters: {best_prophet_params}")

# ==========================================
# 6. HYPERPARAMETER TUNING - XGBOOST
# ==========================================
print("\n" + "="*60)
print("PHASE 2: HYPERPARAMETER TUNING - XGBOOST")
print("="*60)

# XGBoost hyperparameter grid
xgb_params = {
    'lags': [7, 14, 21, 30],
    'lags_past_covariates': [7, 14, 21, 30],
    'output_chunk_length': [7],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

best_xgb_mape = float('inf')
best_xgb_params = None
best_xgb_pred = None

print(f"Testing {len(xgb_params['lags']) * len(xgb_params['lags_past_covariates']) * len(xgb_params['n_estimators']) * len(xgb_params['max_depth']) * len(xgb_params['learning_rate'])} combinations...")

iteration = 0
for lags, lags_cov, n_est, depth, lr in product(
    xgb_params['lags'],
    xgb_params['lags_past_covariates'],
    xgb_params['n_estimators'],
    xgb_params['max_depth'],
    xgb_params['learning_rate']
):
    iteration += 1
    try:
        model = XGBModel(
            lags=lags,
            lags_past_covariates=lags_cov,
            output_chunk_length=7,
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            random_state=42
        )
        model.fit(series=train_scaled, past_covariates=train_vol_scaled)
        pred_scaled = model.predict(len(val_orig), past_covariates=vol_series_scaled)
        pred = price_scaler.inverse_transform(pred_scaled)
        current_mape = mape(val_orig, pred)
        
        if current_mape < best_xgb_mape:
            best_xgb_mape = current_mape
            best_xgb_params = {
                'lags': lags,
                'lags_past_covariates': lags_cov,
                'n_estimators': n_est,
                'max_depth': depth,
                'learning_rate': lr
            }
            best_xgb_pred = pred
            print(f"✓ Iteration {iteration}: New best MAPE = {current_mape:.4f}% | Params: {best_xgb_params}")
    except Exception as e:
        print(f"✗ Iteration {iteration} failed: {e}")
        continue

x_mape = mape(val_orig, best_xgb_pred)
x_rmse = rmse(val_orig, best_xgb_pred)

print(f"\n Best XGBoost MAPE: {x_mape:.2f}%")
print(f" Best XGBoost RMSE: ${x_rmse:.2f}")
print(f"Best Parameters: {best_xgb_params}")

# ==========================================
# 7. ENSEMBLE TUNING (FIND OPTIMAL WEIGHTS)
# ==========================================
print("\n" + "="*60)
print("PHASE 3: ENSEMBLE WEIGHT TUNING")
print("="*60)

# Now tune the ensemble weights using the BEST predictions from each model
# This ensures unbiased ensemble tuning

# Create a grid of weight combinations
# We'll test prophet_factor from 0.0 to 1.0, and xgb_factor = 1 - prophet_factor
weight_steps = np.linspace(0.0, 1.0, 21)  # 21 steps from 0.0 to 1.0

best_ensemble_mape = float('inf')
best_prophet_factor = None
best_xgb_factor = None
best_ensemble_pred = None

print(f"Testing {len(weight_steps)} weight combinations...")
print("Prophet Factor | XGBoost Factor | Ensemble MAPE")
print("-" * 50)

for prophet_factor in weight_steps:
    xgb_factor = 1.0 - prophet_factor
    
    # Combine predictions using weighted average
    # We need to combine the two TimeSeries objects
    ensemble_values = (prophet_factor * best_prophet_pred.values() + 
                      xgb_factor * best_xgb_pred.values())
    
    # Create a new TimeSeries with the ensemble predictions
    ensemble_pred = TimeSeries.from_times_and_values(
        times=best_prophet_pred.time_index,
        values=ensemble_values
    )
    
    # Calculate MAPE for this ensemble
    current_mape = mape(val_orig, ensemble_pred)
    
    print(f"     {prophet_factor:.2f}      |      {xgb_factor:.2f}       |   {current_mape:.4f}%")
    
    if current_mape < best_ensemble_mape:
        best_ensemble_mape = current_mape
        best_prophet_factor = prophet_factor
        best_xgb_factor = xgb_factor
        best_ensemble_pred = ensemble_pred

ensemble_rmse = rmse(val_orig, best_ensemble_pred)

print("\n" + "="*60)
print(" BEST ENSEMBLE WEIGHTS FOUND!")
print("="*60)
print(f"Prophet Factor: {best_prophet_factor:.3f}")
print(f"XGBoost Factor: {best_xgb_factor:.3f}")
print(f"Ensemble MAPE: {best_ensemble_mape:.2f}%")
print(f"Ensemble RMSE: ${ensemble_rmse:.2f}")

# ==========================================
# 8. FINAL COMPARISON
# ==========================================
print("\n" + "="*60)
print("FINAL PERFORMANCE COMPARISON (ON VALIDATION SET)")
print("="*60)
print(f"Prophet (Tuned)  -> MAPE: {p_mape:.2f}% | RMSE: ${p_rmse:.2f}")
print(f"XGBoost (Tuned)  -> MAPE: {x_mape:.2f}% | RMSE: ${x_rmse:.2f}")
print(f"Ensemble (Tuned) -> MAPE: {best_ensemble_mape:.2f}% | RMSE: ${ensemble_rmse:.2f}")

if best_ensemble_mape < p_mape and best_ensemble_mape < x_mape:
    print("\n Ensemble model outperforms both individual models!")
elif p_mape < x_mape:
    print(f"\n Prophet is the winner (but ensemble helps: {best_ensemble_mape:.2f}% vs {p_mape:.2f}%)")
else:
    print(f"\n XGBoost is the winner (but ensemble helps: {best_ensemble_mape:.2f}% vs {x_mape:.2f}%)")

# ==========================================
# 9. FINAL FORECAST (NEXT 3 DAYS)
# ==========================================
print("\n" + "="*60)
print("FINAL FORECAST - NEXT 3 DAYS (Retrained on Full Data)")
print("="*60)

# Retrain best Prophet on full data
final_prophet = Prophet(**best_prophet_params)
final_prophet.fit(price_series)
future_pred_prophet = final_prophet.predict(3)

# Retrain best XGBoost on full data
final_xgb = XGBModel(
    lags=best_xgb_params['lags'],
    lags_past_covariates=best_xgb_params['lags_past_covariates'],
    output_chunk_length=3,
    n_estimators=best_xgb_params['n_estimators'],
    max_depth=best_xgb_params['max_depth'],
    learning_rate=best_xgb_params['learning_rate'],
    random_state=42
)
final_xgb.fit(series=price_series_scaled, past_covariates=vol_series_scaled)
future_pred_xgb_scaled = final_xgb.predict(3, past_covariates=vol_series_scaled)
future_pred_xgb = price_scaler.inverse_transform(future_pred_xgb_scaled)

# Create ensemble prediction for next 3 days
future_ensemble_values = (best_prophet_factor * future_pred_prophet.values() + 
                         best_xgb_factor * future_pred_xgb.values())
future_pred_ensemble = TimeSeries.from_times_and_values(
    times=future_pred_prophet.time_index,
    values=future_ensemble_values
)

print(f"\nProphet Prediction (Next 3 Days):  {future_pred_prophet.values().flatten()}")
print(f"XGBoost Prediction (Next 3 Days):  {future_pred_xgb.values().flatten()}")
print(f"Ensemble Prediction (Next 3 Days): {future_pred_ensemble.values().flatten()}")
print(f"\nEnsemble Formula: {best_prophet_factor:.3f} × Prophet + {best_xgb_factor:.3f} × XGBoost")

# ==========================================
# 10. PLOTTING
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# -------- SUBPLOT 1: Full Comparison --------
ax1.set_title('NVDA: Model Comparison (Tuned)', fontsize=14, fontweight='bold')

# Plot History (last 60 days)
price_series[-60:].plot(ax=ax1, label='Actual History', linewidth=2)

# Plot Validation
val_orig.plot(ax=ax1, label='Actual (Validation)', color='black', linestyle='-', linewidth=2.5)
best_prophet_pred.plot(ax=ax1, label=f'Prophet Val (MAPE: {p_mape:.2f}%)', linestyle='--', linewidth=2, alpha=0.7)
best_xgb_pred.plot(ax=ax1, label=f'XGBoost Val (MAPE: {x_mape:.2f}%)', linestyle=':', linewidth=2, alpha=0.7)
best_ensemble_pred.plot(ax=ax1, label=f'Ensemble Val (MAPE: {best_ensemble_mape:.2f}%)', linestyle='-.', linewidth=2.5, color='purple')

# Plot Future Predictions
future_pred_prophet.plot(ax=ax1, label='Prophet Forecast', color='blue', linestyle='--', linewidth=2, alpha=0.6)
future_pred_xgb.plot(ax=ax1, label='XGBoost Forecast', color='orange', linestyle=':', linewidth=2, alpha=0.6)
future_pred_ensemble.plot(ax=ax1, label='Ensemble Forecast', color='purple', linestyle='-', linewidth=3)

ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Price ($)', fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# -------- SUBPLOT 2: Ensemble Weights Visualization --------
ax2.set_title('Ensemble Weight Performance', fontsize=14, fontweight='bold')

# Calculate MAPE for all weight combinations (for plotting)
mape_values = []
prophet_factors = []
for prophet_factor in weight_steps:
    xgb_factor = 1.0 - prophet_factor
    ensemble_values = (prophet_factor * best_prophet_pred.values() + 
                      xgb_factor * best_xgb_pred.values())
    ensemble_pred = TimeSeries.from_times_and_values(
        times=best_prophet_pred.time_index,
        values=ensemble_values
    )
    mape_values.append(mape(val_orig, ensemble_pred))
    prophet_factors.append(prophet_factor)

ax2.plot(prophet_factors, mape_values, linewidth=2, color='purple')
ax2.axvline(x=best_prophet_factor, color='red', linestyle='--', linewidth=2, 
            label=f'Optimal: Prophet={best_prophet_factor:.2f}, XGB={best_xgb_factor:.2f}')
ax2.axhline(y=best_ensemble_mape, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax2.scatter([best_prophet_factor], [best_ensemble_mape], color='red', s=100, zorder=5)

ax2.set_xlabel('Prophet Factor (XGBoost Factor = 1 - Prophet Factor)', fontsize=11)
ax2.set_ylabel('Ensemble MAPE (%)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# ==========================================
# 11. SAVE BEST PARAMETERS
# ==========================================
print("\n" + "="*60)
print("SAVING BEST PARAMETERS")
print("="*60)

results = {
    'prophet_best_params': best_prophet_params,
    'prophet_mape': float(p_mape),
    'prophet_rmse': float(p_rmse),
    'xgboost_best_params': best_xgb_params,
    'xgboost_mape': float(x_mape),
    'xgboost_rmse': float(x_rmse),
    'ensemble_weights': {
        'prophet_factor': float(best_prophet_factor),
        'xgb_factor': float(best_xgb_factor)
    },
    'ensemble_mape': float(best_ensemble_mape),
    'ensemble_rmse': float(ensemble_rmse),
    'winner': 'Ensemble' if best_ensemble_mape < min(p_mape, x_mape) else ('Prophet' if p_mape < x_mape else 'XGBoost'),
    'forecast_next_3_days': {
        'prophet': future_pred_prophet.values().flatten().tolist(),
        'xgboost': future_pred_xgb.values().flatten().tolist(),
        'ensemble': future_pred_ensemble.values().flatten().tolist()
    }
}

import json
with open('tuning_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✓ Results saved to 'tuning_results.json'")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Model: {results['winner']}")
print(f"Improvement over baseline:")
print(f"  - Prophet alone: {p_mape:.2f}% MAPE")
print(f"  - XGBoost alone: {x_mape:.2f}% MAPE")
print(f"  - Ensemble: {best_ensemble_mape:.2f}% MAPE")
print(f"\nEnsemble uses: {best_prophet_factor:.1%} Prophet + {best_xgb_factor:.1%} XGBoost")
print("\nDone! ")