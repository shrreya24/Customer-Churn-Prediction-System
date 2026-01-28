"""SHAP analysis for model explainability."""
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class SHAPAnalyzer:
    """SHAP-based explainability for churn models."""
    
    def __init__(self, model, X_background: pd.DataFrame,
                 config: Optional[Config] = None):
        """Initialize SHAP analyzer.
        
        Args:
            model: Trained model (tree-based or linear)
            X_background: Background dataset for SHAP
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.model = model
        self.X_background = X_background
        self.outputs_path = Path(config.outputs.get('figures_path', 'outputs/figures/'))
        self.shap_samples = config.explainability.get('shap_samples', 100)
        self.explainer = None
        self.shap_values = None
        self.feature_names = X_background.columns.tolist()
        
    def create_explainer(self):
        """Create SHAP explainer based on model type."""
        print("Creating SHAP explainer...")
        
        # Sample background data if too large
        if len(self.X_background) > self.shap_samples:
            background_sample = self.X_background.sample(n=self.shap_samples, random_state=42)
        else:
            background_sample = self.X_background
        
        # Try tree explainer first (fastest for tree models)
        try:
            self.explainer = shap.TreeExplainer(self.model, background_sample)
            print("✓ Using TreeExplainer (optimized for tree-based models)")
        except:
            # Fall back to KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                background_sample
            )
            print("✓ Using KernelExplainer (model-agnostic)")
        
        return self.explainer
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for given data.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            self.create_explainer()
        
        print(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get values for class 1 (churn)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        self.shap_values = shap_values
        
        print("✓ SHAP values computed")
        
        return shap_values
    
    def plot_summary(self, X: pd.DataFrame,
                    shap_values: Optional[np.ndarray] = None,
                    max_display: int = 20,
                    save_path: Optional[str] = None):
        """Plot SHAP summary showing global feature importance.
        
        Args:
            X: Features
            shap_values: Pre-computed SHAP values (optional)
            max_display: Maximum features to display
            save_path: Optional path to save figure
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title('SHAP Summary Plot - Global Feature Importance',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP summary plot saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_waterfall(self, X: pd.DataFrame, index: int = 0,
                      shap_values: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
        """Plot SHAP waterfall for individual prediction.
        
        Args:
            X: Features
            index: Index of sample to explain
            shap_values: Pre-computed SHAP values (optional)
            save_path: Optional path to save figure
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Create explanation object
        if self.explainer is None:
            self.create_explainer()
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # For binary classification
        
        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=base_value,
            data=X.iloc[index].values,
            feature_names=X.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP waterfall plot saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_force(self, X: pd.DataFrame, index: int = 0,
                  shap_values: Optional[np.ndarray] = None):
        """Plot SHAP force plot for individual prediction.
        
        Args:
            X: Features
            index: Index of sample to explain
            shap_values: Pre-computed SHAP values (optional)
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        if self.explainer is None:
            self.create_explainer()
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        shap.force_plot(
            base_value,
            shap_values[index],
            X.iloc[index],
            matplotlib=True
        )
    
    def plot_dependence(self, X: pd.DataFrame, feature: str,
                       shap_values: Optional[np.ndarray] = None,
                       interaction_feature: Optional[str] = 'auto',
                       save_path: Optional[str] = None):
        """Plot SHAP dependence showing feature interactions.
        
        Args:
            X: Features
            feature: Feature name to plot
            shap_values: Pre-computed SHAP values (optional)
            interaction_feature: Feature for interaction coloring
            save_path: Optional path to save figure
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, shap_values, X,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP dependence plot saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def get_top_features(self, X: pd.DataFrame,
                        shap_values: Optional[np.ndarray] = None,
                        top_n: int = 10) -> pd.DataFrame:
        """Get top features by mean absolute SHAP value.
        
        Args:
            X: Features
            shap_values: Pre-computed SHAP values (optional)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features and importance
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0,
                          top_n: int = 5) -> Dict:
        """Get detailed explanation for a single prediction.
        
        Args:
            X: Features
            index: Index of sample to explain
            top_n: Number of top contributing features
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get SHAP values for this sample
        sample_shap = self.shap_values[index]
        sample_features = X.iloc[index]
        
        # Get top positive and negative contributors
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'value': sample_features.values,
            'shap_value': sample_shap
        })
        
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        top_contributors = shap_df.head(top_n)
        
        explanation = {
            'sample_index': index,
            'top_features': top_contributors.to_dict('records'),
            'base_value': self.explainer.expected_value,
            'predicted_value': self.explainer.expected_value + sample_shap.sum()
        }
        
        return explanation


def main():
    """Main function to demonstrate SHAP analysis."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    from src.models.tree_models import TreeBasedModel
    
    print("=" * 60)
    print("SHAP Explainability Analysis")
    print("=" * 60)
    
    # Load and prepare data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    # Clean and engineer features
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Encode and prepare
    df_encoded = preprocessor.encode_features(df_features, fit=True)
    X, y = preprocessor.prepare_features_target(df_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train a model (use XGBoost for best SHAP support)
    print("\nTraining XGBoost model for SHAP analysis...")
    model = TreeBasedModel(model_type='xgboost')
    model.train(X_train, y_train)
    
    # Create SHAP analyzer
    print("\n" + "=" * 60)
    print("Creating SHAP Analyzer")
    print("=" * 60)
    
    shap_analyzer = SHAPAnalyzer(
        model=model.model,
        X_background=X_train
    )
    
    # Compute SHAP values for test set
    shap_values = shap_analyzer.compute_shap_values(X_test.head(100))
    
    # Generate visualizations
    print("\n1. SHAP Summary Plot")
    shap_analyzer.plot_summary(X_test.head(100), shap_values,
                               save_path='shap_summary.png')
    
    print("\n2. SHAP Waterfall for Sample Prediction")
    shap_analyzer.plot_waterfall(X_test.head(100), index=0, shap_values=shap_values,
                                 save_path='shap_waterfall_example.png')
    
    print("\n3. Top Features by SHAP Importance")
    top_features = shap_analyzer.get_top_features(X_test.head(100), shap_values)
    print(top_features)
    
    print("\n✓ SHAP analysis complete!")
    
    return shap_analyzer


if __name__ == "__main__":
    main()
