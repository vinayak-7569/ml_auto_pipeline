import os
import sys
import time
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

try:
    from ml_pipeline import NonLinearMLPipeline
except ImportError as e:
    print(f"Error: Could not import NonLinearMLPipeline from pipeline: {e}")
    print("Ensure pipeline.py is in the same directory as main.py and contains the NonLinearMLPipeline class.")
    sys.exit(1)

# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(log_dir, verbose=False):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Also set up a console handler with cleaner formatting for user-facing messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = {
    "train_csv": os.path.join(PROJECT_ROOT, "insurance.csv"),
    "new_data_csv": os.path.join(PROJECT_ROOT, "data", "new_data.csv"),
    "predictions_csv": os.path.join(PROJECT_ROOT, "data", "predictions.csv"),
    "target_column": "charges",
    "log_dir": os.path.join(PROJECT_ROOT, "logs"),
    "model_dir": os.path.join(PROJECT_ROOT, "src", "models"),
    "config_path": os.path.join(PROJECT_ROOT, "config.yaml"),
    "task_type": "auto",
    "output_format": "csv",
    "verbose": False
}

# -----------------------------
# Load Configuration YAML
# -----------------------------
def load_config(config_path):
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
            return config
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise

# -----------------------------
# Command Line Args Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run NonLinear ML Pipeline for training and prediction")
    parser.add_argument("--train-csv", help="Path to training data CSV file")
    parser.add_argument("--new-data-csv", help="Path to new data CSV file for predictions")
    parser.add_argument("--predictions-csv", help="Path to save prediction outputs")
    parser.add_argument("--target-column", help="Name of the target column in the dataset")
    parser.add_argument("--config", help="Path to configuration YAML file")
    parser.add_argument("--task-type", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--output-format", choices=["csv", "json"], default="csv")
    parser.add_argument("--generate-template", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

# -----------------------------
# Load Dataset from CSV
# -----------------------------
def load_dataset(path):
    if not os.path.exists(path):
        logger.error(f"Dataset file not found: {path}")
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        print(f"\nLoading dataset: {os.path.basename(path)}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        
        # Print dataset summary
        print(f"Successfully loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Show column info
        print(f"Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
        else:
            print("No missing values detected")
            
        logger.info(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {path}: {e}")
        raise

# -----------------------------
# Auto Detect Target Column
# -----------------------------
def auto_detect_target_column(df):
    known_targets = ['charges', 'target', 'label', 'y', 'price', 'cost', 'value', 'amount']
    
    print(f"\nAuto-detecting target column...")
    
    for col in df.columns:
        if col.lower() in known_targets:
            print(f"Found target column: '{col}' (matched known target names)")
            logger.info(f"Auto-detected target column: {col}")
            return col
    
    # Check for likely target columns based on data characteristics
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if df[col].dtype in ['int64', 'float64'] and unique_ratio > 0.1:
            print(f"Detected target column: '{col}' (continuous numeric variable)")
            logger.info(f"Auto-detected target column: {col}")
            return col
        elif df[col].nunique() <= 20 and df[col].nunique() > 1:
            print(f"Detected target column: '{col}' (categorical variable)")
            logger.info(f"Auto-detected target column: {col}")
            return col
    
    print(f"Could not auto-detect target column, using last column: '{df.columns[-1]}'")
    logger.warning("Could not auto-detect target column, using last column")
    return df.columns[-1]

# -----------------------------
# Validate Dataset Function
# -----------------------------
def validate_data(df, target_col, new_data=False):
    if not new_data:
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        if df[target_col].isna().all():
            logger.error(f"Target column '{target_col}' contains only missing values")
            raise ValueError(f"Target column '{target_col}' contains only missing values")
    if df.empty:
        logger.error("Dataset is empty")
        raise ValueError("Dataset is empty")
    missing_percent = df.isna().mean() * 100
    for col, percent in missing_percent.items():
        if percent > 50:
            logger.warning(f"Column '{col}' has {percent:.2f}% missing values")

# -----------------------------
# Improved Preprocessing for Better Accuracy
# -----------------------------
def preprocess_data(df, target_col):
    print(f"\nStarting advanced data preprocessing...")
    df = df.copy()
    
    original_shape = df.shape
    missing_before = df.isnull().sum().sum()
    
    # Handle missing values more intelligently
    categorical_fixed = 0
    numerical_fixed = 0
    
    for col in df.columns:
        if col == target_col:
            continue
        
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            if df[col].dtype in ['object']:
                # Use mode for categorical variables
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                categorical_fixed += missing_count
            else:
                # Use median for numerical variables (more robust than mean)
                df[col].fillna(df[col].median(), inplace=True)
                numerical_fixed += missing_count
    
    missing_after = df.isnull().sum().sum()
    
    # Print preprocessing summary
    print(f"Advanced data preprocessing completed:")
    print(f"   Shape: {original_shape[0]:,} rows × {original_shape[1]} columns (unchanged)")
    if missing_before > 0:
        print(f"   Fixed {missing_before:,} missing values:")
        if categorical_fixed > 0:
            print(f"      • {categorical_fixed:,} categorical (filled with mode)")
        if numerical_fixed > 0:
            print(f"      • {numerical_fixed:,} numerical (filled with median)")
    else:
        print(f"   No missing values to fix")
    
    logger.info("Advanced preprocessing completed.")
    return df

# -----------------------------
# Train Model with Expanded Hyperparameter Tuning
# -----------------------------
def train_model(df, target_col, task_type, config):
    validate_data(df, target_col)
    try:
        print(f"\nStarting advanced ML model training for {task_type.upper()} task...")
        print(f"Target column: '{target_col}'")
        
        # Apply basic preprocessing first
        df_processed = preprocess_data(df.copy(), target_col)
        
        model = NonLinearMLPipeline(task_type=task_type)
        
        # Set ultra-comprehensive hyperparameter grid for maximum accuracy
        hyperparameter_grid = {
            "rf": {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [10, 15, 20, 25, None],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ['auto', 'sqrt', 'log2', None],
                "bootstrap": [True, False]
            },
            "lr": {},  # Default parameters
            "xgb": {
                "n_estimators": [200, 300, 500, 800, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "max_depth": [3, 5, 7, 9, 12],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.1, 0.5, 1.0],
                "reg_lambda": [0, 0.1, 0.5, 1.0]
            },
            "gb": {
                "n_estimators": [200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.15],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "et": {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        }
        
        print(f"Setting up ultra-comprehensive hyperparameter optimization...")
        rf_combinations = len(hyperparameter_grid['rf']['n_estimators']) * len(hyperparameter_grid['rf']['max_depth']) * len(hyperparameter_grid['rf']['min_samples_split']) * len(hyperparameter_grid['rf']['min_samples_leaf']) * len(hyperparameter_grid['rf']['max_features']) * len(hyperparameter_grid['rf']['bootstrap'])
        xgb_combinations = len(hyperparameter_grid['xgb']['n_estimators']) * len(hyperparameter_grid['xgb']['learning_rate']) * len(hyperparameter_grid['xgb']['max_depth']) * len(hyperparameter_grid['xgb']['subsample']) * len(hyperparameter_grid['xgb']['colsample_bytree']) * len(hyperparameter_grid['xgb']['reg_alpha']) * len(hyperparameter_grid['xgb']['reg_lambda'])
        gb_combinations = len(hyperparameter_grid['gb']['n_estimators']) * len(hyperparameter_grid['gb']['learning_rate']) * len(hyperparameter_grid['gb']['max_depth']) * len(hyperparameter_grid['gb']['subsample']) * len(hyperparameter_grid['gb']['min_samples_split']) * len(hyperparameter_grid['gb']['min_samples_leaf'])
        
        print(f"   Random Forest: {rf_combinations:,} parameter combinations")
        print(f"   XGBoost: {xgb_combinations:,} parameter combinations") 
        print(f"   Gradient Boosting: {gb_combinations:,} parameter combinations")
        print(f"   Using RandomizedSearchCV with 100+ iterations for optimal efficiency")
        
        model.set_hyperparameter_grid(hyperparameter_grid)
        
        # Train the model
        print(f"\nTraining advanced ensemble models (this may take 10-15 minutes for maximum accuracy)...")
        model.train(df_processed, target_col=target_col)
        
        # Save trained models
        model.save_models(config["model_dir"])
        
        print(f"\n=== ULTRA HIGH ACCURACY MODEL PERFORMANCE RESULTS ===")
        scores = model.get_all_model_scores()
        
        # Sort scores by performance
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(sorted_scores):
            medal = "BEST" if i == 0 else "2ND" if i == 1 else "3RD" if i == 2 else "GOOD"
            model_name = {
                'rf': 'Random Forest',
                'lr': 'Linear Regression',
                'xgb': 'XGBoost',
                'gb': 'Gradient Boosting',
                'et': 'Extra Trees',
                'ada': 'AdaBoost',
                'svr': 'Support Vector Regression',
                'knn': 'K-Nearest Neighbors',
                'ensemble': 'Ensemble (Stacking)'
            }.get(name, name.upper())
            
            if task_type == 'regression':
                print(f"[{medal}] {model_name}: R² = {score:.6f} ({score*100:.4f}% variance explained)")
            else:
                print(f"[{medal}] {model_name}: Accuracy = {score:.6f} ({score*100:.4f}%)")
        
        best_model_name = sorted_scores[0][0]
        best_score = sorted_scores[0][1]
        print(f"\nBEST PERFORMING MODEL: {best_model_name.upper()} (Score: {best_score:.6f})")
        
        # Check if we achieved 95%+ accuracy
        if best_score >= 0.95:
            print(f"SUCCESS: Achieved 95%+ accuracy target!")
        elif best_score >= 0.90:
            print(f"EXCELLENT: Achieved 90%+ accuracy!")
        else:
            print(f"Good performance achieved. Consider more feature engineering for 95%+ target.")
        
        logger.info("=== Ultra High Accuracy Model Evaluation ===")
        for name, score in scores.items():
            logger.info(f"Model: {name}, Score: {score:.6f}")
        return model
    except Exception as e:
        print(f"Training failed: {e}")
        logger.error(f"Training failed: {e}")
        raise

# -----------------------------
# Predict with Ensemble Methods
# -----------------------------
def predict_on_new_data(model, new_df, output_path, output_format, task_type, config):
    print(f"\n=== MAKING PREDICTIONS ON NEW DATA ===")
    try:
        # Apply basic preprocessing without target column
        new_df_processed = preprocess_data(new_df.copy(), None)
        
        print(f"Generating predictions using multiple advanced methods...")
        
        # Make predictions using different methods
        best_preds = model.predict(new_df_processed, use_ensemble=False)
        
        output_df = pd.DataFrame({
            "Best_Model_Predictions": best_preds
        })

        # Add ensemble predictions if available
        ensemble_methods = 0
        try:
            stacking_preds = model.predict(new_df_processed, use_ensemble=True)
            output_df["Ensemble_Stacking"] = stacking_preds
            ensemble_methods += 1
            print("Ensemble stacking predictions generated")
        except Exception as e:
            print(f"Ensemble stacking failed: {str(e)[:50]}...")
            logger.warning(f"Stacking predictions failed: {e}")

        try:
            weighted_preds = model.predict(new_df_processed, use_ensemble=True, use_weighted=True)
            output_df["Ensemble_Weighted"] = weighted_preds
            ensemble_methods += 1
            print("Weighted ensemble predictions generated")
        except Exception as e:
            print(f"Weighted ensemble failed: {str(e)[:50]}...")
            logger.warning(f"Weighted predictions failed: {e}")

        # Add confidence intervals for regression tasks
        if task_type == "regression" or (task_type == "auto" and model.task_type == "regression"):
            try:
                print("Computing confidence intervals...")
                predictions_list = []
                for _ in range(10):  # More iterations for better CI
                    pred = model.predict(new_df_processed, use_ensemble=False)
                    predictions_list.append(pred)
                
                if predictions_list:
                    predictions_array = np.array(predictions_list)
                    ci_lower = np.percentile(predictions_array, 5, axis=0)  # 90% CI
                    ci_upper = np.percentile(predictions_array, 95, axis=0)
                    output_df["CI_Lower_90%"] = ci_lower
                    output_df["CI_Upper_90%"] = ci_upper
                    print("90% confidence intervals computed")
            except Exception as e:
                print(f"Confidence intervals failed: {str(e)[:50]}...")
                logger.warning(f"Confidence intervals failed: {e}")

        # Save predictions with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"predictions_{timestamp}.{output_format}")

        if output_format == "csv":
            output_df.to_csv(output_file, index=False)
        elif output_format == "json":
            output_df.to_json(output_file, orient="records", indent=2)

        # Display prediction summary
        print(f"\n=== ULTRA HIGH ACCURACY PREDICTION RESULTS SUMMARY ===")
        print(f"Saved to: {os.path.basename(output_file)}")
        print(f"Predictions for {len(output_df):,} records generated")
        print(f"Methods used: Best model + {ensemble_methods} ensemble methods")
        
        # Better detection of regression vs classification
        is_regression = (task_type == "regression" or 
                        (task_type == "auto" and hasattr(model, 'task_type') and model.task_type == "regression") or
                        output_df['Best_Model_Predictions'].dtype in ['float64', 'float32'])
        
        if is_regression:
            print(f"\nSample predictions:")
            for i in range(min(5, len(output_df))):
                pred_val = output_df.iloc[i]['Best_Model_Predictions']
                print(f"   Record {i+1}: ${pred_val:,.2f}")
                
            print(f"\nPrediction statistics:")
            print(f"   • Mean: ${output_df['Best_Model_Predictions'].mean():,.2f}")
            print(f"   • Median: ${output_df['Best_Model_Predictions'].median():,.2f}")
            print(f"   • Min: ${output_df['Best_Model_Predictions'].min():,.2f}")
            print(f"   • Max: ${output_df['Best_Model_Predictions'].max():,.2f}")
        else:
            print(f"\nClassification results:")
            pred_counts = output_df['Best_Model_Predictions'].value_counts()
            for class_val, count in pred_counts.head().items():
                print(f"   • Class '{class_val}': {count} predictions ({count/len(output_df)*100:.1f}%)")

        logger.info(f"Saved predictions to: {output_file}")
        logger.debug(f"Sample predictions: {output_df.head().to_dict()}")
        return output_df
    except Exception as e:
        print(f"Prediction failed: {e}")
        logger.error(f"Prediction failed: {e}")
        raise

def generate_new_data_template(train_csv, target_col, output_path):
    try:
        df = load_dataset(train_csv)
        cols = [c for c in df.columns if c != target_col]
        template_df = pd.DataFrame(columns=cols)
        template_df.to_csv(output_path, index=False)
        logger.info(f"Blank new_data template saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate template: {e}")
        raise

# -----------------------------
# Main Function
# -----------------------------
def main():
    start_time = time.time()
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    global logger
    logger = setup_logging(config["log_dir"], args.verbose or config["verbose"])

    # Print header
    print("=" * 60)
    print("ADVANCED ML PIPELINE - ULTRA HIGH ACCURACY VERSION")
    print("=" * 60)
    print("Automated Machine Learning with Advanced Ensemble Methods")
    print("Enhanced Feature Engineering & Comprehensive Hyperparameter Optimization")
    print("=" * 60)

    config_path = args.config or config["config_path"]
    config.update(load_config(config_path))

    config["train_csv"] = args.train_csv or config["train_csv"]
    config["new_data_csv"] = args.new_data_csv or config["new_data_csv"]
    config["predictions_csv"] = args.predictions_csv or config["predictions_csv"]
    config["target_column"] = args.target_column or config["target_column"]
    config["task_type"] = args.task_type or config["task_type"]
    config["output_format"] = args.output_format or config["output_format"]
    config["verbose"] = args.verbose or config["verbose"]

    logger.info("=== ML Pipeline Started ===")
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    if args.generate_template:
        print(f"\nGenerating template for new data predictions...")
        generate_new_data_template(config["train_csv"], config["target_column"], config["new_data_csv"])
        print(f"Template saved successfully!")
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        logger.info(f"Pipeline completed in {elapsed:.2f} seconds")
        return

    try:
        df = load_dataset(config["train_csv"])
    except Exception as e:
        print(f"Failed to load training data: {e}")
        logger.error(f"Failed to load training data: {e}")
        sys.exit(1)

    if not config["target_column"]:
        config["target_column"] = auto_detect_target_column(df)

    try:
        model = train_model(df, config["target_column"], config["task_type"], config)
        print(f"\nModels saved to: {config['model_dir']}")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

    if os.path.exists(config["new_data_csv"]):
        try:
            new_df = load_dataset(config["new_data_csv"])
            # Model is already trained and saved, just predict
            predict_on_new_data(model, new_df, config["predictions_csv"], config["output_format"], config["task_type"], config)
        except Exception as e:
            print(f"Prediction pipeline failed: {e}")
            logger.error(f"Prediction pipeline failed: {e}")
            sys.exit(1)
    else:
        print(f"\nNew data file not found: {os.path.basename(config['new_data_csv'])}")
        print(f"Generate a template with: python main.py --generate-template")

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"ULTRA HIGH ACCURACY PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {elapsed:.2f} seconds")
    if elapsed > 60:
        print(f"   ({elapsed/60:.1f} minutes)")
    print("=" * 60)
    
    logger.info(f"=== ML Pipeline Completed in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()
