"""Evaluation utilities for model performance assessment."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score,
    recall_score, precision_score, f1_score
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 threshold: float = 0.5):
        """Initialize evaluator with predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = (y_pred_proba >= threshold).astype(int)
        self.threshold = threshold
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'roc_auc': roc_auc_score(self.y_true, self.y_pred_proba),
            'average_precision': average_precision_score(self.y_true, self.y_pred_proba),
            'recall': recall_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'f1_score': f1_score(self.y_true, self.y_pred),
        }
        return metrics
    
    def plot_roc_curve(self, ax: Optional[plt.Axes] = None, 
                       label: str = 'Model') -> plt.Axes:
        """Plot ROC curve.
        
        Args:
            ax: Matplotlib axes object
            label: Label for the curve
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_precision_recall_curve(self, ax: Optional[plt.Axes] = None,
                                    label: str = 'Model') -> plt.Axes:
        """Plot Precision-Recall curve.
        
        Args:
            ax: Matplotlib axes object
            label: Label for the curve
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        ax.plot(recall, precision, label=f'{label} (AP = {avg_precision:.3f})', 
                linewidth=2)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_confusion_matrix(self, ax: Optional[plt.Axes] = None,
                             normalize: bool = False) -> plt.Axes:
        """Plot confusion matrix.
        
        Args:
            ax: Matplotlib axes object
            normalize: Whether to normalize values
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax, cbar=True)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_yticklabels(['No Churn', 'Churn'])
        
        return ax
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string
        """
        return classification_report(self.y_true, self.y_pred,
                                     target_names=['No Churn', 'Churn'])
    
    def plot_probability_distribution(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot distribution of predicted probabilities by class.
        
        Args:
            ax: Matplotlib axes object
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate probabilities by true class
        churn_probs = self.y_pred_proba[self.y_true == 1]
        no_churn_probs = self.y_pred_proba[self.y_true == 0]
        
        ax.hist(no_churn_probs, bins=50, alpha=0.6, label='No Churn', 
                color='blue', density=True)
        ax.hist(churn_probs, bins=50, alpha=0.6, label='Churn',
                color='red', density=True)
        ax.axvline(self.threshold, color='black', linestyle='--',
                   label=f'Threshold ({self.threshold})')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Probability Distribution by True Class', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax


def compare_models(evaluators: Dict[str, ModelEvaluator],
                   save_path: Optional[str] = None):
    """Compare multiple models with comprehensive visualizations.
    
    Args:
        evaluators: Dictionary of model name to ModelEvaluator
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ROC curves
    for name, evaluator in evaluators.items():
        evaluator.plot_roc_curve(ax=axes[0, 0], label=name)
    
    # Precision-Recall curves
    for name, evaluator in evaluators.items():
        evaluator.plot_precision_recall_curve(ax=axes[0, 1], label=name)
    
    # Metrics comparison
    metrics_df = pd.DataFrame({
        name: evaluator.compute_metrics()
        for name, evaluator in evaluators.items()
    }).T
    
    metrics_df.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_xlabel('Model', fontsize=12)
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Table of metrics
    axes[1, 1].axis('off')
    table_data = metrics_df.round(3).reset_index()
    table = axes[1, 1].table(cellText=table_data.values,
                            colLabels=table_data.columns,
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Detailed Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
