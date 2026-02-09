"""
Visualization module for model evaluation and insights
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MarketplaceVisualizer:
    """Create visualizations for marketplace revenue optimization"""
    
    def __init__(self, figsize: Tuple = (10, 6)):
        self.figsize = figsize
        
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str], n_cols: int = 3):
        """Plot distributions of important features"""
        
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], 3 * n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            if feature in df.columns:
                ax = axes[idx]
                if df[feature].dtype in ['int64', 'float64']:
                    df[feature].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_title(f'Distribution of {feature}', fontsize=10)
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Frequency')
                else:
                    df[feature].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f'Distribution of {feature}', fontsize=10)
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Count')
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple = (12, 10)):
        """Plot correlation heatmap of numerical features"""
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True,
                       cbar_kws={"shrink": 0.8})
            
            plt.title('Feature Correlation Heatmap', fontsize=14, pad=20)
            plt.tight_layout()
            plt.show()
    
    def plot_model_performance_comparison(self, results: Dict):
        """Plot comparison of model performances"""
        
        models = list(results.keys())
        auc_scores = [results[model]['best_score'] for model in models]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(models, auc_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        
        # Add value labels
        for bar, score in zip(bars, auc_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=10)
        
        ax.set_xlabel('ROC-AUC Score')
        ax.set_title('Model Performance Comparison (Cross-Validation AUC)')
        ax.set_xlim([0, max(auc_scores) * 1.2])
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot precision-recall curve"""
        
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=self.figsize)
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return thresholds, precision, recall
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot ROC curve"""
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fpr, tpr, thresholds
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15):
        """Plot feature importance"""
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # Highest importance on top
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{imp:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix_heatmap(self, cm: np.ndarray, classes: List[str] = ['Low', 'High']):
        """Plot confusion matrix as heatmap"""
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_business_impact_analysis(self, metrics: Dict, baseline_metrics: Dict = None):
        """Plot business impact metrics"""
        
        if baseline_metrics is None:
            baseline_metrics = {
                'precision': 0.7,
                'recall': 0.65,
                'f1_score': 0.675,
                'roc_auc': 0.72
            }
        
        # Calculate improvements
        metrics_list = ['precision', 'recall', 'f1_score', 'roc_auc']
        improvements = [
            (metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric] * 100 
            for metric in metrics_list
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Metric values
        x = range(len(metrics_list))
        axes[0].bar(x, [metrics[metric] for metric in metrics_list], 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Metrics')
        axes[0].grid(True, alpha=0.3)
        
        # Add baseline line
        axes[0].axhline(y=baseline_metrics['precision'], color='blue', linestyle='--', alpha=0.5)
        axes[0].axhline(y=baseline_metrics['recall'], color='green', linestyle='--', alpha=0.5)
        axes[0].axhline(y=baseline_metrics['f1_score'], color='orange', linestyle='--', alpha=0.5)
        axes[0].axhline(y=baseline_metrics['roc_auc'], color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Improvements
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1].bar(x, improvements, color=colors, alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Improvement Over Baseline')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(axes[1].patches, improvements)):
            axes[1].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (1 if imp > 0 else -1),
                        f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_price_rating_plot(self, df: pd.DataFrame, target: pd.Series):
        """Create interactive scatter plot of price vs rating colored by conversion"""
        
        fig = px.scatter(
            df, x='price', y='rating', color=target.astype(str),
            color_discrete_map={'0': 'red', '1': 'green'},
            title='Price vs Rating (Green = High Conversion, Red = Low Conversion)',
            labels={'color': 'Conversion'},
            opacity=0.6,
            hover_data=['discount_pct', 'review_count'] if 'discount_pct' in df.columns else None
        )
        
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Rating (1-5)",
            legend_title="Conversion"
        )
        
        return fig