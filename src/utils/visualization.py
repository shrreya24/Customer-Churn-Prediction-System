"""Visualization utilities for consistent plotting."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def plot_churn_distribution(df: pd.DataFrame, churn_col: str = 'Churn',
                            save_path: Optional[str] = None):
    """Plot distribution of churn labels.
    
    Args:
        df: DataFrame containing churn data
        churn_col: Name of churn column
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    churn_counts = df[churn_col].value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values, 
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
    axes[0].set_xlabel('Churn Status', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Churn', 'Churn'])
    
    # Add value labels on bars
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    # Percentage plot
    churn_pct = df[churn_col].value_counts(normalize=True) * 100
    axes[1].bar(churn_pct.index, churn_pct.values,
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
    axes[1].set_xlabel('Churn Status', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Churn', 'Churn'])
    
    # Add percentage labels
    for i, v in enumerate(churn_pct.values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str],
                               churn_col: str = 'Churn',
                               save_path: Optional[str] = None):
    """Plot distributions of features by churn status.
    
    Args:
        df: DataFrame containing data
        features: List of feature names to plot
        churn_col: Name of churn column
        save_path: Optional path to save figure
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        if df[feature].dtype in ['object', 'category']:
            # Categorical feature
            pd.crosstab(df[feature], df[churn_col], normalize='index').plot(
                kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.8
            )
            ax.set_ylabel('Proportion', fontsize=10)
            ax.legend(['No Churn', 'Churn'], loc='upper right')
        else:
            # Numerical feature
            df[df[churn_col] == 0][feature].hist(ax=ax, bins=30, alpha=0.6,
                                                  color='#2ecc71', label='No Churn')
            df[df[churn_col] == 1][feature].hist(ax=ax, bins=30, alpha=0.6,
                                                  color='#e74c3c', label='Churn')
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend()
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str],
                             save_path: Optional[str] = None):
    """Plot correlation heatmap for numerical features.
    
    Args:
        df: DataFrame containing data
        features: List of feature names
        save_path: Optional path to save figure
    """
    # Select only numerical features
    numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_features) == 0:
        print("No numerical features to plot correlation")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = df[numerical_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_churn_by_segments(df: pd.DataFrame, segment_col: str,
                           churn_col: str = 'Churn',
                           save_path: Optional[str] = None):
    """Plot churn rate by segments.
    
    Args:
        df: DataFrame containing data
        segment_col: Column name for segmentation
        churn_col: Name of churn column
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Churn rate by segment
    churn_by_segment = df.groupby(segment_col)[churn_col].agg(['mean', 'count'])
    churn_by_segment = churn_by_segment.sort_values('mean', ascending=False)
    
    axes[0].barh(range(len(churn_by_segment)), churn_by_segment['mean'] * 100,
                 color='#e74c3c', alpha=0.8)
    axes[0].set_yticks(range(len(churn_by_segment)))
    axes[0].set_yticklabels(churn_by_segment.index)
    axes[0].set_xlabel('Churn Rate (%)', fontsize=12)
    axes[0].set_ylabel(segment_col, fontsize=12)
    axes[0].set_title(f'Churn Rate by {segment_col}', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(churn_by_segment['mean'] * 100):
        axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Count by segment
    axes[1].barh(range(len(churn_by_segment)), churn_by_segment['count'],
                 color='#3498db', alpha=0.8)
    axes[1].set_yticks(range(len(churn_by_segment)))
    axes[1].set_yticklabels(churn_by_segment.index)
    axes[1].set_xlabel('Customer Count', fontsize=12)
    axes[1].set_ylabel(segment_col, fontsize=12)
    axes[1].set_title(f'Customer Count by {segment_col}', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(churn_by_segment['count']):
        axes[1].text(v + 50, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
