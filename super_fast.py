#!/usr/bin/env python3
"""
SUPER FAST ML Pipeline - Under 10 seconds guaranteed
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

def main():
    print("SUPER FAST ML Pipeline - Maximum Speed (Under 10 seconds)")
    start_time = time.time()
    
    # Load and preprocess in one go
    df = pd.read_csv("insurance.csv")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Lightning fast preprocessing
    df.columns = df.columns.str.lower()
    
    # Vectorized missing value handling (faster)
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    
    # Only the most critical features for speed
    df['age_bmi'] = df['age'] * df['bmi']
    df['smoker_binary'] = (df['smoker'] == 'yes').astype(int)
    df['smoker_age'] = df['smoker_binary'] * df['age']
    df['smoker_bmi'] = df['smoker_binary'] * df['bmi']
    
    # Fast encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Split
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    print(f"Features: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Ultra-fast optimized model
    model = RandomForestRegressor(
        n_estimators=50,   # Minimal trees for maximum speed
        max_depth=10,      # Shallow trees for speed
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training super-fast model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"R² Score: {r2:.6f} ({r2*100:.4f}% accuracy)")
    
    # Success metrics
    if r2 >= 0.90:
        print("EXCELLENT: Achieved 90%+ accuracy!")
    elif r2 >= 0.85:
        print("VERY GOOD: Achieved 85%+ accuracy!")
    elif r2 >= 0.80:
        print("GOOD: Achieved 80%+ accuracy!")
    else:
        print(f"Achieved {r2*100:.2f}% accuracy")
    
    # Quick predictions
    print("Sample predictions:")
    for i in range(3):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(f"  Actual: ${actual:.0f}, Predicted: ${predicted:.0f}")
    
    elapsed = time.time() - start_time
    print(f"SUPER FAST completed in {elapsed:.3f} seconds!")

if __name__ == "__main__":
    main()
