"""
Data generation module for synthetic marketplace data
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class MarketplaceDataGenerator:
    """Generate synthetic marketplace data with realistic patterns"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def generate_product_features(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate product-level features"""
        
        # Base features
        data = {
            'product_id': [f'PROD_{i:05d}' for i in range(n_samples)],
            'category': np.random.choice(
                ['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 
                n_samples,
                p=[0.25, 0.2, 0.25, 0.15, 0.15]
            ),
            'price': np.random.uniform(10, 500, n_samples),
            'original_price': np.random.uniform(15, 600, n_samples),
            'discount_pct': np.random.uniform(0, 0.7, n_samples),
        }
        
        # Calculate derived pricing features
        df = pd.DataFrame(data)
        df['price_ratio'] = df['price'] / df['original_price']
        df['is_discounted'] = (df['discount_pct'] > 0.1).astype(int)
        
        return df
    
    def generate_rating_features(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate rating and review features"""
        
        data = {
            'rating': np.random.beta(5, 2, n_samples) * 4 + 1,  # Beta distribution for ratings
            'review_count': np.random.poisson(50, n_samples),
            'rating_std': np.random.exponential(0.5, n_samples),
            'positive_review_pct': np.random.beta(8, 2, n_samples),
            'recent_review_count': np.random.poisson(10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add interaction features
        df['rating_review_interaction'] = df['rating'] * np.log1p(df['review_count'])
        df['rating_confidence'] = df['rating'] / (1 + df['rating_std'])
        
        return df
    
    def generate_temporal_features(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate temporal features"""
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2024-01-01')
        
        dates = pd.date_range(start_date, end_date, periods=n_samples)
        
        data = {
            'listing_date': dates,
            'days_listed': np.random.exponential(100, n_samples),
            'season': pd.cut(dates.month, 
                           bins=[0, 3, 6, 9, 12], 
                           labels=['Winter', 'Spring', 'Summer', 'Fall']),
            'is_weekend_listed': (dates.dayofweek >= 5).astype(int),
        }
        
        return pd.DataFrame(data)
    
    def generate_target(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate target variable based on features (high-conversion = 1)"""
        
        # Simulate conversion probability based on features
        conversion_prob = (
            0.3 * (features_df['discount_pct'] > 0.2) +
            0.4 * (features_df['rating'] > 3.5) +
            0.2 * (features_df['price'] < 200) +
            0.1 * (features_df['review_count'] > 20)
        )
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(features_df))
        conversion_prob = np.clip(conversion_prob + noise, 0, 1)
        
        # Convert to binary target
        target = (conversion_prob > 0.5).astype(int)
        
        # Ensure class balance
        positive_ratio = target.mean()
        if positive_ratio < 0.4 or positive_ratio > 0.6:
            print(f"Adjusting target distribution: {positive_ratio:.2%}")
            threshold = np.percentile(conversion_prob, 50)
            target = (conversion_prob > threshold).astype(int)
        
        return target
    
    def generate_complete_dataset(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate complete dataset with all features and target"""
        
        print(f"Generating synthetic marketplace data ({n_samples} samples)...")
        
        # Generate all feature groups
        product_features = self.generate_product_features(n_samples)
        rating_features = self.generate_rating_features(n_samples)
        temporal_features = self.generate_temporal_features(n_samples)
        
        # Combine features
        features_df = pd.concat([product_features, rating_features, temporal_features], axis=1)
        
        # Generate target
        target = self.generate_target(features_df)
        
        print(f"Dataset created: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        print(f"Target distribution: {target.mean():.2%} high-conversion products")
        
        return features_df, target