#!/usr/bin/env python3
"""
Ultra-fast ML pipeline optimized for maximum speed and accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

def main():
    print("ULTRA FAST ML Pipeline - Maximum Speed & Accuracy")
    start_time = time.time()
    
    # Load and preprocess in one go
    df = pd.read_csv("insurance.csv")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Lightning fast preprocessing
    df.columns = df.columns.str.lower()
    
    # Vectorized missing value handling (faster)
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    
    # Smart feature engineering (only the most impactful)
    df['age_bmi'] = df['age'] * df['bmi']
    df['smoker_age'] = (df['smoker'] == 'yes').astype(int) * df['age']
    df['smoker_bmi'] = (df['smoker'] == 'yes').astype(int) * df['bmi']
    df['high_bmi'] = (df['bmi'] > 30).astype(int)
    df['senior'] = (df['age'] > 50).astype(int)
    
    # Fast encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Split
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    print(f"Features: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Ultra-optimized model
    model = RandomForestRegressor(
        n_estimators=150,  # Sweet spot for speed vs accuracy
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',  # Faster feature selection
        random_state=42,
        n_jobs=-1
    )
    
    print("Training ultra-fast model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"R² Score: {r2:.6f} ({r2*100:.4f}% accuracy)")
    
    # Success metrics
    if r2 >= 0.95:
        print("SUCCESS: Achieved 95%+ accuracy!")
    elif r2 >= 0.90:
        print("EXCELLENT: Achieved 90%+ accuracy!")
    elif r2 >= 0.85:
        print("GOOD: Achieved 85%+ accuracy!")
    else:
        print(f"Achieved {r2*100:.2f}% accuracy")
    
    # Quick sample
    sample_preds = model.predict(X_test[:3])
    sample_actual = y_test[:3].values
    print("Sample predictions:")
    for i in range(3):
        print(f"  Actual: ${sample_actual[i]:.0f}, Predicted: ${sample_preds[i]:.0f}")
    
    elapsed = time.time() - start_time
    print(f"ULTRA FAST completed in {elapsed:.3f} seconds!")

if __name__ == "__main__":
    main()
