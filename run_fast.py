#!/usr/bin/env python3
"""
Fast version of the ML pipeline for quick testing
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time

def main():
    print("LIGHTNING FAST ML Pipeline - Maximum Speed & Accuracy")
    start_time = time.time()
    
    # Load data
    data_path = "insurance.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Ultra-fast preprocessing with vectorized operations
    df.columns = df.columns.str.lower()
    
    # Vectorized missing value handling (much faster)
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    
    # Only the most impactful features for maximum speed
    df['age_bmi'] = df['age'] * df['bmi']
    df['smoker_binary'] = (df['smoker'] == 'yes').astype(int)
    df['smoker_age'] = df['smoker_binary'] * df['age']
    df['smoker_bmi'] = df['smoker_binary'] * df['bmi']
    df['high_bmi'] = (df['bmi'] > 30).astype(int)
    
    # Fast encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Split features and target
    target_col = 'charges'
    if target_col not in df_encoded.columns:
        print(f"Target column '{target_col}' not found!")
        return
        
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    print(f"Features: {X.shape[1]}, Target: {target_col}")
    
    # Train-test split with better validation for higher accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Ultra-fast single model training
    print("Training lightning-fast optimized model...")
    
    # Use only the best performing model for maximum speed
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced for speed
        max_depth=12,      # Reduced for speed
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("  Training Random Forest (Optimized for Speed)...")
    model.fit(X_train, y_train)
    
    # Quick evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"    R² Score: {r2:.6f} ({r2*100:.4f}% accuracy)")
    
    best_model = model
    best_score = r2
    
    print(f"\nBest model: Random Forest (Optimized for Speed)")
    print(f"Best R² Score: {best_score:.6f} ({best_score*100:.4f}% accuracy)")
    
    # Check if we achieved 95%+ accuracy
    if best_score >= 0.95:
        print("SUCCESS: Achieved 95%+ accuracy target!")
    elif best_score >= 0.90:
        print("EXCELLENT: Achieved 90%+ accuracy!")
    elif best_score >= 0.85:
        print("GOOD: Achieved 85%+ accuracy!")
    else:
        print(f"Achieved {best_score*100:.2f}% accuracy")
    
    # Feature importance for insights
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X.columns
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nTop 3 most important features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
    
    # Quick sample predictions
    print(f"\nSample predictions:")
    sample_preds = best_model.predict(X_test[:3])
    sample_actual = y_test[:3].values
    
    for i in range(3):
        print(f"  Actual: ${sample_actual[i]:.0f}, Predicted: ${sample_preds[i]:.0f}")
    
    elapsed = time.time() - start_time
    print(f"\nLIGHTNING FAST Pipeline completed in {elapsed:.2f} seconds!")
    print("=" * 60)

if __name__ == "__main__":
    main()
