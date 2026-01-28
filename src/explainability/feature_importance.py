"""Feature importance extraction and visualization."""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from typing import Optional, List, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize feature importance analyzer.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.outputs_path = Path(config.outputs.get('figures_path', 'outputs/figures/'))
    
    def get_model_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract built-in feature importance from model.
        
        Args:
            model: Trained model with feature_importances_ or coef_
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            importance_type = 'gini_importance'
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
            importance_type = 'coefficient'
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_type': importance_type
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 10,
                                   random_state: int = 42) -> pd.DataFrame:
        """Calculate permutation importance (model-agnostic).
        
        Args:
            model: Trained model
            X: Features
            y: Target
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            DataFrame with permutation importance
        """
        print(f"Computing permutation importance (n_repeats={n_repeats})...")
        
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        print("✓ Permutation importance computed")
        
        return importance_df
    
    def plot_importance_comparison(self, importance_dfs: Dict[str, pd.DataFrame],
                                   top_n: int = 15,
                                   save_path: Optional[str] = None):
        """Plot comparison of feature importance from different methods.
        
        Args:
            importance_dfs: Dictionary of method name to importance DataFrame
            top_n: Number of top features to show
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, len(importance_dfs), figsize=(6 * len(importance_dfs), 8))
        
        if len(importance_dfs) == 1:
            axes = [axes]
        
        for idx, (method, df) in enumerate(importance_dfs.items()):
            ax = axes[idx]
            
            # Get top N features
            df_top = df.head(top_n).sort_values('importance', ascending=True)
            
            # Plot horizontal bar chart
            colors = sns.color_palette("viridis", len(df_top))
            ax.barh(range(len(df_top)), df_top['importance'], color=colors)
            ax.set_yticks(range(len(df_top)))
            ax.set_yticklabels(df_top['feature'])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'{method}\nTop {top_n} Features',
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance comparison saved to {full_path}")
        
        return fig
    
    def plot_partial_dependence(self, model, X: pd.DataFrame,
                               features: List[str],
                               save_path: Optional[str] = None):
        """Plot partial dependence plots.
        
        Args:
            model: Trained model
            X: Features
            features: List of features to plot (max 4 recommended)
            save_path: Optional path to save figure
        """
        print(f"Computing partial dependence for {len(features)} features...")
        
        # Get feature indices
        feature_indices = [X.columns.get_loc(f) for f in features if f in X.columns]
        
        fig, ax = plt.subplots(figsize=(15, 4 * ((len(feature_indices) + 1) // 2)))
        
        display = PartialDependenceDisplay.from_estimator(
            model, X, feature_indices,
            ax=ax, n_cols=2, grid_resolution=50
        )
        
        plt.suptitle('Partial Dependence Plots', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Partial dependence plot saved to {full_path}")
        
        return fig


def main():
    """Main function to demonstrate feature importance analysis."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    from src.models.tree_models import TreeBasedModel
    from src.models.baseline import BaselineModel
    
    print("=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    # Load and prepare data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    df_encoded = preprocessor.encode_features(df_features, fit=True)
    X, y = preprocessor.prepare_features_target(df_encoded)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    print("\n1. Training Random Forest...")
    rf_model = TreeBasedModel(model_type='random_forest')
    rf_model.train(X_train, y_train)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Get different types of importance
    print("\n2. Extracting Feature Importance")
    
    # Built-in importance
    rf_importance = analyzer.get_model_importance(rf_model.model, X_train.columns.tolist())
    rf_importance.rename(columns={'importance': 'importance'}, inplace=True)
    
    # Permutation importance
    perm_importance = analyzer.get_permutation_importance(
        rf_model.model, X_test, y_test, n_repeats=5
    )
    perm_importance.rename(columns={'importance_mean': 'importance'}, inplace=True)
    
    # Plot comparison
    print("\n3. Visualizing Feature Importance")
    importance_dfs = {
        'Random Forest (Gini)': rf_importance,
        'Permutation Importance': perm_importance
    }
    
    analyzer.plot_importance_comparison(
        importance_dfs,
        save_path='feature_importance_comparison.png'
    )
    
    # Partial dependence plots for top features
    print("\n4. Partial Dependence Plots")
    top_features = rf_importance.head(4)['feature'].tolist()
    analyzer.plot_partial_dependence(
        rf_model.model, X_test,
        features=top_features,
        save_path='partial_dependence.png'
    )
    
    print("\n✓ Feature importance analysis complete!")
    
    return analyzer


if __name__ == "__main__":
    main()
