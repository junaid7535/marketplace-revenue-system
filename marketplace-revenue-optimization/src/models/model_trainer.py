"""
Model training module for marketplace revenue optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..features.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Train multiple models for marketplace conversion prediction"""
    
    def __init__(self, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_engineer = FeatureEngineer()
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize model dictionary with configurations"""
        
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Prepare data for training with feature engineering"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Apply feature engineering
        print("Applying feature engineering...")
        self.feature_engineer.fit(X_train)
        X_train_processed = self.feature_engineer.transform(X_train)
        X_test_processed = self.feature_engineer.transform(X_test)
        
        print(f"Training set: {X_train_processed.shape}")
        print(f"Test set: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models using cross-validation"""
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\nTraining {name}...")
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=self.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"  Best AUC: {grid_search.best_score_:.4f}")
            print(f"  Best params: {grid_search.best_params_}")
        
        self.results = results
        return results
    
    def select_best_model(self) -> Tuple[Any, str]:
        """Select the best performing model"""
        
        best_model_name = max(
            self.results.items(), 
            key=lambda x: x[1]['best_score']
        )[0]
        
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\n{'='*50}")
        print(f"Best Model: {best_model_name}")
        print(f"Validation AUC: {self.results[best_model_name]['best_score']:.4f}")
        print(f"{'='*50}")
        
        return self.best_model, best_model_name
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate best model on test set"""
        
        if self.best_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Save model and feature engineer
        model_data = {
            'model': self.best_model,
            'feature_engineer': self.feature_engineer,
            'results': self.results
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.results = model_data.get('results', {})
        
        print(f"Model loaded from {filepath}")
        
        return self.best_model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model"""
        
        if self.best_model is None:
            raise ValueError("No model selected.")
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.feature_engineer.get_feature_names()
            
            # If no feature names, create generic ones
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            print("Model doesn't support feature importance directly.")
            return pd.DataFrame()