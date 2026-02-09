"""
Feature engineering module for marketplace data
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for marketplace data"""
    
    def __init__(self, create_interactions: bool = True, scale_features: bool = True):
        self.create_interactions = create_interactions
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self.numerical_features = None
        self.categorical_features = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Identify feature types"""
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.scale_features and self.numerical_features:
            self.scaler.fit(X[self.numerical_features])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        X_transformed = X.copy()
        
        # 1. Create derived features
        X_transformed = self._create_derived_features(X_transformed)
        
        # 2. Create interaction features
        if self.create_interactions:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # 3. Handle categorical features
        X_transformed = self._encode_categorical_features(X_transformed)
        
        # 4. Scale numerical features
        if self.scale_features and self.numerical_features:
            X_transformed[self.numerical_features] = self.scaler.transform(
                X_transformed[self.numerical_features]
            )
        
        # 5. Remove any remaining NaN values
        X_transformed = X_transformed.fillna(0)
        
        return X_transformed
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones"""
        
        # Pricing features
        if all(col in df.columns for col in ['price', 'original_price']):
            df['price_to_original_ratio'] = df['price'] / df['original_price']
            df['absolute_discount'] = df['original_price'] - df['price']
            df['discount_effectiveness'] = df['discount_pct'] * np.log1p(df['review_count'])
        
        # Rating features
        if all(col in df.columns for col in ['rating', 'review_count']):
            df['rating_weighted'] = df['rating'] * np.log1p(df['review_count'])
            df['rating_confidence_score'] = df['rating'] * (1 - df['rating_std'] / 5)
            df['review_sentiment_score'] = df['rating'] * df['positive_review_pct']
        
        # Temporal features
        if 'days_listed' in df.columns:
            df['listing_recency'] = 1 / (1 + df['days_listed'])
            df['seasonal_adjusted_price'] = df['price'] * self._get_seasonal_multiplier(df)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        # Price-Rating interactions
        if all(col in df.columns for col in ['price', 'rating']):
            df['price_rating_interaction'] = df['price'] * df['rating']
            df['value_for_money'] = df['rating'] / df['price']
        
        # Discount-Rating interactions
        if all(col in df.columns for col in ['discount_pct', 'rating']):
            df['discounted_quality'] = df['discount_pct'] * df['rating']
        
        # Temporal-Rating interactions
        if all(col in df.columns for col in ['days_listed', 'rating']):
            df['aged_rating'] = df['rating'] / np.log1p(df['days_listed'])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        # One-hot encode categories
        if 'category' in df.columns:
            categories = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df.drop('category', axis=1), categories], axis=1)
        
        if 'season' in df.columns:
            seasons = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df.drop('season', axis=1), seasons], axis=1)
        
        return df
    
    def _get_seasonal_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """Calculate seasonal adjustment multiplier"""
        seasonal_multipliers = {
            'Winter': 1.1,  # Higher prices in winter
            'Spring': 0.95,
            'Summer': 0.9,  # Lower prices in summer
            'Fall': 1.05
        }
        
        if 'season' in df.columns:
            return df['season'].map(seasonal_multipliers).fillna(1.0)
        return pd.Series(1.0, index=df.index)
    
    def get_feature_names(self) -> list:
        """Get list of feature names after transformation"""
        if hasattr(self, 'feature_names_'):
            return self.feature_names_
        return []