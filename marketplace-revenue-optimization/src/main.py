"""
Main execution script for Marketplace Revenue Optimization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import custom modules
from data.data_generator import MarketplaceDataGenerator
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from evaluation.visualizations import MarketplaceVisualizer
from evaluation.metrics import calculate_business_metrics


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("MARKETPLACE REVENUE OPTIMIZATION - ML MODEL")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. GENERATING SYNTHETIC MARKETPLACE DATA")
    print("-" * 40)
    
    data_gen = MarketplaceDataGenerator(seed=42)
    X, y = data_gen.generate_complete_dataset(n_samples=5000)
    
    # Step 2: Initialize model trainer
    print("\n2. INITIALIZING MODEL TRAINER")
    print("-" * 40)
    
    trainer = ModelTrainer(
        test_size=0.2,
        cv_folds=5,
        random_state=42
    )
    
    # Step 3: Prepare data
    print("\n3. PREPARING DATA FOR TRAINING")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Step 4: Train models
    print("\n4. TRAINING MODELS WITH CROSS-VALIDATION")
    print("-" * 40)
    
    results = trainer.train_models(X_train, y_train)
    
    # Step 5: Select best model
    print("\n5. SELECTING BEST MODEL")
    print("-" * 40)
    
    best_model, best_model_name = trainer.select_best_model()
    
    # Step 6: Evaluate on test set
    print("\n6. EVALUATING ON TEST SET")
    print("-" * 40)
    
    test_metrics = trainer.evaluate_on_test(X_test, y_test)
    
    # Step 7: Calculate business metrics
    print("\n7. CALCULATING BUSINESS METRICS")
    print("-" * 40)
    
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    business_metrics = calculate_business_metrics(y_test, y_pred_proba)
    
    print(f"Expected Revenue Lift: {business_metrics['expected_revenue_lift']:.2%}")
    print(f"Conversion Rate Improvement: {business_metrics['conversion_improvement']:.2%}")
    print(f"Precision at Top 20%: {business_metrics['precision_at_k']:.4f}")
    
    # Step 8: Generate visualizations
    print("\n8. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    visualizer = MarketplaceVisualizer()
    
    # Plot model performance comparison
    visualizer.plot_model_performance_comparison(results)
    
    # Plot ROC curve
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    visualizer.plot_roc_curve(y_test.values, y_pred_proba)
    
    # Plot precision-recall curve
    visualizer.plot_precision_recall_curve(y_test.values, y_pred_proba)
    
    # Plot feature importance
    importance_df = trainer.get_feature_importance()
    if not importance_df.empty:
        visualizer.plot_feature_importance(importance_df, top_n=15)
    
    # Plot business impact
    baseline_metrics = {
        'precision': 0.70,
        'recall': 0.65,
        'f1_score': 0.675,
        'roc_auc': 0.72
    }
    visualizer.plot_business_impact_analysis(test_metrics, baseline_metrics)
    
    # Step 9: Save model
    print("\n9. SAVING MODEL")
    print("-" * 40)
    
    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = artifacts_dir / "best_marketplace_model.pkl"
    trainer.save_model(str(model_path))
    
    # Step 10: Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['roc_auc']:.4f}")
    print(f"Model Improvement Over Baseline: ~{((test_metrics['roc_auc']/0.72)-1)*100:.0f}%")
    
    print("\nKey Features Used:")
    if not importance_df.empty:
        top_features = importance_df.head(5)['feature'].tolist()
        for i, feature in enumerate(top_features, 1):
            print(f"  {i}. {feature}")
    
    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT READY")
    print("=" * 60)


if __name__ == "__main__":
    main()