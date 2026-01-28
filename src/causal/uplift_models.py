"""Causal inference and uplift modeling for intervention analysis."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class UpliftModel:
    """Uplift modeling for causal intervention analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize uplift model.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.causal_config = config.causal
        self.treatment_features = self.causal_config.get('treatment_features', [])
        self.outputs_path = Path(config.outputs.get('figures_path', 'outputs/figures/'))
        self.models = {}
        self.uplift_scores = {}
    
    def create_treatment_variable(self, df: pd.DataFrame,
                                  treatment_feature: str) -> pd.DataFrame:
        """Create binary treatment variable from feature.
        
        Args:
            df: DataFrame with features
            treatment_feature: Name of treatment feature
            
        Returns:
            DataFrame with treatment column
        """
        df_treat = df.copy()
        
        if treatment_feature == 'TechSupport':
            # Treatment: Has tech support
            if df_treat[treatment_feature].dtype == 'object':
                df_treat['treatment'] = (df_treat[treatment_feature] == 'Yes').astype(int)
            else:
                df_treat['treatment'] = df_treat[treatment_feature]
        
        elif treatment_feature == 'Contract':
            # Treatment: Long-term contract (1-year or 2-year)
            df_treat['treatment'] = df_treat[treatment_feature].isin(['One year', 'Two year']).astype(int)
        
        elif treatment_feature == 'PaperlessBilling':
            # Treatment: Paperless billing
            if df_treat[treatment_feature].dtype == 'object':
                df_treat['treatment'] = (df_treat[treatment_feature] == 'Yes').astype(int)
            else:
                df_treat['treatment'] = df_treat[treatment_feature]
        
        else:
            # General binary treatment
            if df_treat[treatment_feature].dtype == 'object':
                df_treat['treatment'] = (df_treat[treatment_feature] == 'Yes').astype(int)
            else:
                df_treat['treatment'] = (df_treat[treatment_feature] > df_treat[treatment_feature].median()).astype(int)
        
        return df_treat
    
    def s_learner(self, X: pd.DataFrame, y: pd.Series,
                 treatment: pd.Series) -> Dict:
        """S-Learner: Single model with treatment as feature.
        
        Args:
            X: Features
            y: Target (churn)
            treatment: Treatment indicator
            
        Returns:
            Dictionary with model and uplift predictions
        """
        print("Training S-Learner...")
        
        # Add treatment as a feature
        X_with_treatment = X.copy()
        X_with_treatment['treatment'] = treatment
        
        # Train single model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_with_treatment, y)
        
        # Predict with and without treatment
        X_treated = X_with_treatment.copy()
        X_treated['treatment'] = 1
        
        X_control = X_with_treatment.copy()
        X_control['treatment'] = 0
        
        p_treated = model.predict_proba(X_treated)[:, 1]
        p_control = model.predict_proba(X_control)[:, 1]
        
        # Uplift = difference in churn probability
        uplift = p_control - p_treated  # Positive uplift means treatment reduces churn
        
        print(f"✓ S-Learner trained. Mean uplift: {uplift.mean():.4f}")
        
        return {
            'model': model,
            'uplift': uplift,
            'p_treated': p_treated,
            'p_control': p_control
        }
    
    def t_learner(self, X: pd.DataFrame, y: pd.Series,
                 treatment: pd.Series) -> Dict:
        """T-Learner: Separate models for treated and control groups.
        
        Args:
            X: Features
            y: Target (churn)
            treatment: Treatment indicator
            
        Returns:
            Dictionary with models and uplift predictions
        """
        print("Training T-Learner...")
        
        # Split into treatment and control groups
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]
        
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        # Train separate models
        model_treated = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_control = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        model_treated.fit(X_treated, y_treated)
        model_control.fit(X_control, y_control)
        
        # Predict on all data with both models
        p_treated = model_treated.predict_proba(X)[:, 1]
        p_control = model_control.predict_proba(X)[:, 1]
        
        # Uplift = difference in churn probability
        uplift = p_control - p_treated
        
        print(f"✓ T-Learner trained. Mean uplift: {uplift.mean():.4f}")
        print(f"  Treated group size: {len(X_treated)}")
        print(f"  Control group size: {len(X_control)}")
        
        return {
            'model_treated': model_treated,
            'model_control': model_control,
            'uplift': uplift,
            'p_treated': p_treated,
            'p_control': p_control
        }
    
    def analyze_intervention(self, df: pd.DataFrame, treatment_feature: str,
                            target_col: str = 'Churn',
                            method: str = 'both') -> Dict:
        """Analyze causal effect of intervention.
        
        Args:
            df: DataFrame with all features
            treatment_feature: Name of treatment feature
            target_col: Name of target column
            method: 's-learner', 't-learner', or 'both'
            
        Returns:
            Dictionary with analysis results
        """
        print("=" * 60)
        print(f"Causal Analysis: {treatment_feature}")
        print("=" * 60)
        
        # Create treatment variable
        df_treat = self.create_treatment_variable(df, treatment_feature)
        treatment = df_treat['treatment']
        
        # Prepare features (exclude treatment feature and target)
        exclude_cols = [target_col, treatment_feature, 'treatment']
        feature_cols = [col for col in df_treat.columns if col not in exclude_cols]
        
        X = df_treat[feature_cols]
        y = df_treat[target_col]
        
        results = {
            'treatment_feature': treatment_feature,
            'treatment_rate': treatment.mean(),
            'churn_rate_treated': y[treatment == 1].mean(),
            'churn_rate_control': y[treatment == 0].mean(),
            'naive_effect': y[treatment == 0].mean() - y[treatment == 1].mean()
        }
        
        print(f"\nTreatment rate: {results['treatment_rate']:.2%}")
        print(f"Churn rate (treated): {results['churn_rate_treated']:.2%}")
        print(f"Churn rate (control): {results['churn_rate_control']:.2%}")
        print(f"Naive effect: {results['naive_effect']:.4f}")
        
        # Train models
        if method in ['s-learner', 'both']:
            results['s_learner'] = self.s_learner(X, y, treatment)
        
        if method in ['t-learner', 'both']:
            results['t_learner'] = self.t_learner(X, y, treatment)
        
        self.models[treatment_feature] = results
        
        return results
    
    def plot_uplift_distribution(self, uplift: np.ndarray,
                                 treatment_name: str,
                                 method_name: str = 'Uplift',
                                 save_path: Optional[str] = None):
        """Plot distribution of uplift scores.
        
        Args:
            uplift: Array of uplift scores
            treatment_name: Name of treatment
            method_name: Name of method (S-learner or T-learner)
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(uplift, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].axvline(uplift.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {uplift.mean():.4f}')
        axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
        axes[0].set_xlabel('Uplift Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Uplift Distribution - {treatment_name}\n({method_name})',
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Sorted uplift
        sorted_uplift = np.sort(uplift)[::-1]
        axes[1].plot(sorted_uplift, linewidth=2, color='#e74c3c')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].fill_between(range(len(sorted_uplift)), sorted_uplift, 0,
                            where=(sorted_uplift > 0), alpha=0.3, color='green',
                            label='Positive Uplift')
        axes[1].fill_between(range(len(sorted_uplift)), sorted_uplift, 0,
                            where=(sorted_uplift < 0), alpha=0.3, color='red',
                            label='Negative Uplift')
        axes[1].set_xlabel('Customer Rank', fontsize=12)
        axes[1].set_ylabel('Uplift Score', fontsize=12)
        axes[1].set_title(f'Sorted Uplift Scores - {treatment_name}',
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Uplift distribution plot saved to {full_path}")
        
        return fig
    
    def get_high_uplift_customers(self, X: pd.DataFrame, uplift: np.ndarray,
                                  top_n: int = 100) -> pd.DataFrame:
        """Identify customers with highest uplift potential.
        
        Args:
            X: Features
            uplift: Uplift scores
            top_n: Number of top customers to return
            
        Returns:
            DataFrame with top customers and their characteristics
        """
        X_with_uplift = X.copy()
        X_with_uplift['uplift'] = uplift
        
        # Get top customers
        top_customers = X_with_uplift.nlargest(top_n, 'uplift')
        
        return top_customers


def main():
    """Main function to demonstrate causal inference."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Causal Inference & Uplift Modeling")
    print("=" * 60)
    
    # Load and prepare data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Initialize uplift model
    uplift_model = UpliftModel()
    
    # Analyze interventions
    treatments = ['TechSupport', 'Contract', 'PaperlessBilling']
    
    for treatment in treatments:
        if treatment in df_features.columns:
            print(f"\n{'=' * 60}")
            print(f"Analyzing: {treatment}")
            print('=' * 60)
            
            # Analyze with both methods
            results = uplift_model.analyze_intervention(
                df_features, treatment, method='both'
            )
            
            # Plot S-learner uplift
            if 's_learner' in results:
                uplift_model.plot_uplift_distribution(
                    results['s_learner']['uplift'],
                    treatment,
                    method_name='S-Learner',
                    save_path=f'uplift_{treatment.lower()}_s_learner.png'
                )
            
            # Plot T-learner uplift
            if 't_learner' in results:
                uplift_model.plot_uplift_distribution(
                    results['t_learner']['uplift'],
                    treatment,
                    method_name='T-Learner',
                    save_path=f'uplift_{treatment.lower()}_t_learner.png'
                )
            
            # Get high uplift customers
            if 't_learner' in results:
                X = df_features[[col for col in df_features.columns
                                if col not in ['Churn', treatment]]]
                top_customers = uplift_model.get_high_uplift_customers(
                    X, results['t_learner']['uplift'], top_n=10
                )
                print(f"\nTop 10 customers by uplift:")
                print(f"Mean uplift for top 10: {top_customers['uplift'].mean():.4f}")
    
    print("\n✓ Causal analysis complete!")
    
    return uplift_model


if __name__ == "__main__":
    main()
