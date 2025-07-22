import pandas as pd
import numpy as np
import logging
import joblib

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# Optional import for LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(level=logging.INFO)


class NonLinearMLPipeline:
    # -------------------------
    # Initialization
    # -------------------------
    def __init__(self, task_type='auto'):
        self.task_type = task_type
        self.models = []
        self.ensemble_model = None
        self.best_model = None
        self.weighted_ensemble = None
        self.label_encoder = None
        self.feature_order = None
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.poly = None
        self.model_scores = {}
        self.param_grid = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'lr': {},  # Default parameters for Linear/Logistic Regression
            'xgb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
        }

    # -------------------------
    # Detect the type of ML task
    # -------------------------
    def detect_task_type(self, y):
        if self.task_type == 'auto':
            if y.dtype == 'object' or y.nunique() <= 20:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        logging.info(f"Detected task type: {self.task_type}")

    # -------------------------
    # Data Preprocessing
    # -------------------------
    def preprocess(self, df, target_col=None):
        df = df.copy()
        if target_col:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df
            y = None

        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        # Numerical preprocessing
        if len(num_cols) > 0:
            # Handle outliers
            for col in num_cols:
                X[col] = np.where(X[col] > X[col].quantile(0.99), X[col].quantile(0.99), X[col])
                X[col] = np.where(X[col] < X[col].quantile(0.01), X[col].quantile(0.01), X[col])

            if self.num_imputer is None:
                self.num_imputer = SimpleImputer(strategy='median')
                X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
            else:
                X[num_cols] = self.num_imputer.transform(X[num_cols])

            if self.scaler is None:
                self.scaler = StandardScaler()
                X[num_cols] = self.scaler.fit_transform(X[num_cols])
            else:
                X[num_cols] = self.scaler.transform(X[num_cols])

            if self.poly is None:
                # Use degree 3 polynomial features for even better accuracy
                self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
                poly_features = pd.DataFrame(
                    self.poly.fit_transform(X[num_cols]),
                    columns=self.poly.get_feature_names_out(num_cols),
                    index=X.index,
                )
            else:
                poly_features = pd.DataFrame(
                    self.poly.transform(X[num_cols]),
                    columns=self.poly.get_feature_names_out(num_cols),
                    index=X.index,
                )

            # Add additional feature engineering for ultra-high accuracy
            # Add log transformations for positive numerical features
            for col in num_cols:
                if X[col].min() > 0:
                    poly_features[f'{col}_log'] = np.log1p(X[col])
                    poly_features[f'{col}_sqrt'] = np.sqrt(X[col])
                    
            X = pd.concat([X.drop(columns=num_cols), poly_features], axis=1)

        # Categorical preprocessing
        if len(cat_cols) > 0:
            if self.cat_imputer is None:
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
                X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])
            else:
                X[cat_cols] = self.cat_imputer.transform(X[cat_cols])

        X = pd.get_dummies(X)

        # Maintain fixed feature order for consistency during prediction
        if self.feature_order is None:
            self.feature_order = X.columns.tolist()
        else:
            for col in self.feature_order:
                if col not in X.columns:
                    X[col] = 0
            for col in list(X.columns):
                if col not in self.feature_order:
                    X.drop(columns=col, inplace=True)
            X = X[self.feature_order]

        # Label encode target variable for classification
        if target_col and self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        return (X, y) if target_col else X

    # -------------------------
    # Ultra-Enhanced Feature Selection for Maximum Accuracy
    # -------------------------
    def select_features(self, model, X_train, y_train, threshold=0.001):  # Even lower threshold for more features
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        
        # Select top features by importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select features with importance above threshold OR top 50 features (whichever is more)
        features_by_threshold = feature_importance_df[feature_importance_df['importance'] > threshold]['feature'].tolist()
        top_features = feature_importance_df.head(min(50, len(X_train.columns)))['feature'].tolist()
        
        selected_features = list(set(features_by_threshold + top_features))
        logging.info(f"Selected {len(selected_features)} features with importance > {threshold} or in top 50")
        
        return selected_features

    # -------------------------
    # Ultra-Comprehensive Hyperparameter Tuning for Maximum Accuracy
    # -------------------------
    def tune_model(self, model, param_grid, X_train, y_train):
        # Use comprehensive CV and better scoring for accuracy
        scoring = 'neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy'
        
        # Limit parameter combinations for very large grids
        total_combinations = 1
        for param_values in param_grid.values():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        # Use RandomizedSearch for large parameter spaces with more iterations
        if total_combinations > 100:
            from sklearn.model_selection import RandomizedSearchCV
            grid_search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=min(200, total_combinations),  # Much more iterations for better accuracy
                cv=10,  # More CV folds for better validation
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=10,  # More CV folds for better validation
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
        grid_search.fit(X_train, y_train)
        logging.info(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}")
        logging.info(f"Best CV score: {grid_search.best_score_:.6f}")
        return grid_search.best_estimator_

    # -------------------------
    # Training Pipeline
    # -------------------------
    def train(self, df, target_col):
        X, y = self.preprocess(df, target_col)
        self.detect_task_type(y)

        # Use FULL dataset for maximum accuracy - no sampling
        logging.info(f"Using full dataset with {len(X)} samples for maximum accuracy")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if self.task_type == 'classification' else None)

        if self.task_type == 'classification':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info("Applied SMOTE for class imbalance correction.")

        rf_model = (
            RandomForestClassifier(n_estimators=100, random_state=42)
            if self.task_type == 'classification'
            else RandomForestRegressor(n_estimators=100, random_state=42)
        )
        selected_features = self.select_features(rf_model, X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        self.feature_order = selected_features

        if self.task_type == 'classification':
            # Enhanced model list for maximum accuracy
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=300, random_state=42)),
            ]
            
            # Add advanced models for maximum accuracy
            try:
                from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
                from sklearn.svm import SVC
                from sklearn.neighbors import KNeighborsClassifier
                # Add more advanced models
                try:
                    from sklearn.neural_network import MLPClassifier
                    base_models.append(('mlp', MLPClassifier(random_state=42, max_iter=1000)))
                    self.param_grid['mlp'] = {
                        'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive']
                    }
                except ImportError:
                    pass
                    
                base_models.extend([
                    ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42)),
                    ('ada', AdaBoostClassifier(n_estimators=200, random_state=42)),
                    ('svc', SVC(probability=True, random_state=42)),
                    ('knn', KNeighborsClassifier())
                ])
                self.param_grid['gb'] = {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
                self.param_grid['ada'] = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.5, 1.0, 1.5, 2.0]
                }
                self.param_grid['svc'] = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
                self.param_grid['knn'] = {
                    'n_neighbors': [3, 5, 7, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            except ImportError:
                pass
            
            self.models = base_models
            
            # Use ensemble for maximum accuracy
            meta_model = XGBClassifier(n_estimators=100, random_state=42)
            self.ensemble_model = StackingClassifier(
                estimators=self.models,
                final_estimator=meta_model,
                cv=5
            )
        else:
            # Enhanced regression models for maximum accuracy
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=300, random_state=42)),
                ('lr', LinearRegression()),
                ('xgb', XGBRegressor(n_estimators=300, random_state=42)),
            ]
            
            # Add advanced regression models for maximum accuracy
            try:
                from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
                from sklearn.svm import SVR
                from sklearn.neighbors import KNeighborsRegressor
                # Add more advanced models
                try:
                    from sklearn.neural_network import MLPRegressor
                    base_models.append(('mlp', MLPRegressor(random_state=42, max_iter=1000)))
                    self.param_grid['mlp'] = {
                        'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive']
                    }
                except ImportError:
                    pass
                    
                base_models.extend([
                    ('gb', GradientBoostingRegressor(n_estimators=300, random_state=42)),
                    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42)),
                    ('ada', AdaBoostRegressor(n_estimators=200, random_state=42)),
                    ('svr', SVR()),
                    ('knn', KNeighborsRegressor())
                ])
                self.param_grid['gb'] = {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
                self.param_grid['et'] = {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
                self.param_grid['ada'] = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.5, 1.0, 1.5, 2.0]
                }
                self.param_grid['svr'] = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
                self.param_grid['knn'] = {
                    'n_neighbors': [3, 5, 7, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            except ImportError:
                pass
            
            self.models = base_models
            
            # Use ensemble for maximum accuracy
            meta_model = XGBRegressor(n_estimators=100, random_state=42)
            self.ensemble_model = StackingRegressor(
                estimators=self.models,
                final_estimator=meta_model,
                cv=5
            )

        # Train individual models with enhanced reporting
        print(f"Training {len(self.models)} advanced individual models...")
        for i, (name, model) in enumerate(self.models):
            model_display_name = {
                'rf': 'Random Forest',
                'lr': 'Linear/Logistic Regression', 
                'xgb': 'XGBoost',
                'gb': 'Gradient Boosting',
                'et': 'Extra Trees',
                'ada': 'AdaBoost',
                'svc': 'Support Vector Machine',
                'svr': 'Support Vector Regression',
                'knn': 'K-Nearest Neighbors'
            }.get(name, name.upper())
            
            print(f"   [{i+1}/{len(self.models)}] Training {model_display_name}...", end=" ")
            
            if name in self.param_grid and self.param_grid[name]:
                # Use comprehensive parameter tuning
                tuned_model = self.tune_model(model, self.param_grid[name], X_train, y_train)
                print("COMPLETED (with hyperparameter optimization)")
            else:
                # Use model with default parameters
                tuned_model = model
                tuned_model.fit(X_train, y_train)
                print("COMPLETED (default parameters)")
                
            self.model_scores[name] = tuned_model.score(X_test, y_test)
            
            # Update the model in the list with the tuned version
            for j, (model_name, _) in enumerate(self.models):
                if model_name == name:
                    self.models[j] = (name, tuned_model)
                    break

        # Train ensemble for maximum accuracy
        if self.ensemble_model:
            try:
                print(f"Training advanced ensemble model (stacking)...")
                self.ensemble_model.fit(X_train, y_train)
                ensemble_score = self.ensemble_model.score(X_test, y_test)
                self.model_scores['ensemble'] = ensemble_score
                print("Ensemble model trained successfully")
                logging.info(f"Ensemble model score: {ensemble_score:.4f}")
            except Exception as e:
                print(f"Ensemble training failed: {str(e)[:50]}...")
                logging.warning(f"Ensemble training failed: {e}")
                self.ensemble_model = None

        # Save the best model (including ensemble if better)
        all_scores = self.model_scores.copy()
        best_model_name = max(all_scores, key=all_scores.get)
        
        if best_model_name == 'ensemble' and self.ensemble_model:
            self.best_model = self.ensemble_model
            logging.info(f"Best model selected: Ensemble (score: {all_scores[best_model_name]:.4f})")
        else:
            self.best_model = next((model for name, model in self.models if name == best_model_name), self.models[0][1])
            logging.info(f"Best model selected: {best_model_name} (score: {all_scores[best_model_name]:.4f})")

    # -------------------------
    # Set Hyperparameter Grid
    # -------------------------
    def set_hyperparameter_grid(self, grid_updates):
        """Update the hyperparameter grid with custom values."""
        for model_name, params in grid_updates.items():
            if model_name in self.param_grid:
                self.param_grid[model_name].update(params)
            else:
                logging.warning(f"Model {model_name} not found in parameter grid")

    # -------------------------
    # Prediction Methods
    # -------------------------
    def predict(self, df, use_ensemble=False, use_weighted=False):
        """Make predictions on new data."""
        X = self.preprocess(df)
        
        if use_ensemble and self.ensemble_model is not None:
            if use_weighted:
                # Weighted ensemble prediction
                predictions = []
                weights = []
                for name, model in self.models:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_scores[name])
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Weighted average
                weighted_pred = np.average(predictions, axis=0, weights=weights)
                return weighted_pred
            else:
                # Use stacking ensemble (often best performing)
                return self.ensemble_model.predict(X)
        else:
            # Use best individual model
            return self.best_model.predict(X)

    # -------------------------
    # Model Persistence
    # -------------------------
    def save_models(self, model_dir=None):
        """Save trained models and preprocessors."""
        if model_dir is None:
            model_dir = "models"
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, os.path.join(model_dir, "best_pipeline.pkl"))
        
        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, os.path.join(model_dir, "ensemble_pipeline.pkl"))
        
        # Save preprocessors and encoders
        if self.label_encoder:
            joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        
        # Save pipeline metadata
        metadata = {
            'task_type': self.task_type,
            'feature_order': self.feature_order,
            'model_scores': self.model_scores
        }
        joblib.dump(metadata, os.path.join(model_dir, "pipeline_metadata.pkl"))
        
        # Save preprocessors
        preprocessors = {
            'scaler': self.scaler,
            'num_imputer': self.num_imputer,
            'cat_imputer': self.cat_imputer,
            'poly': self.poly
        }
        joblib.dump(preprocessors, os.path.join(model_dir, "preprocessors.pkl"))
        
        logging.info(f"Models saved to {model_dir}")

    def load_models(self, model_dir=None):
        """Load trained models and preprocessors."""
        if model_dir is None:
            model_dir = "models"
        
        import os
        
        # Load best model
        best_model_path = os.path.join(model_dir, "best_pipeline.pkl")
        if os.path.exists(best_model_path):
            self.best_model = joblib.load(best_model_path)
        
        # Load ensemble model
        ensemble_model_path = os.path.join(model_dir, "ensemble_pipeline.pkl")
        if os.path.exists(ensemble_model_path):
            self.ensemble_model = joblib.load(ensemble_model_path)
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "pipeline_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.task_type = metadata['task_type']
            self.feature_order = metadata['feature_order']
            self.model_scores = metadata['model_scores']
        
        # Load preprocessors
        preprocessors_path = os.path.join(model_dir, "preprocessors.pkl")
        if os.path.exists(preprocessors_path):
            preprocessors = joblib.load(preprocessors_path)
            self.scaler = preprocessors.get('scaler')
            self.num_imputer = preprocessors.get('num_imputer')
            self.cat_imputer = preprocessors.get('cat_imputer')
            self.poly = preprocessors.get('poly')
        
        logging.info(f"Models loaded from {model_dir}")

    # -------------------------
    # Get Model Scores
    # -------------------------
    def get_all_model_scores(self):
        """Return all model scores."""
        return self.model_scores.copy()
