"""Survival analysis for time-to-churn estimation."""
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class SurvivalAnalysis:
    """Survival analysis for customer churn."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize survival analysis.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.survival_config = config.survival
        self.time_column = self.survival_config.get('time_column', 'tenure')
        self.event_column = self.survival_config.get('event_column', 'Churn')
        self.outputs_path = Path(config.outputs.get('figures_path', 'outputs/figures/'))
        self.kmf = KaplanMeierFitter()
        self.coxph = CoxPHFitter()
        
    def fit_kaplan_meier(self, df: pd.DataFrame,
                        group_col: Optional[str] = None) -> Dict:
        """Fit Kaplan-Meier survival curves.
        
        Args:
            df: DataFrame with time and event columns
            group_col: Optional column for grouping (e.g., Contract type)
            
        Returns:
            Dictionary with survival data
        """
        print("=" * 60)
        print("Kaplan-Meier Survival Analysis")
        print("=" * 60)
        
        results = {}
        
        if group_col is None:
            # Overall survival curve
            self.kmf.fit(
                durations=df[self.time_column],
                event_observed=df[self.event_column],
                label='Overall'
            )
            results['overall'] = {
                'survival_function': self.kmf.survival_function_,
                'median_survival': self.kmf.median_survival_time_,
                'confidence_interval': self.kmf.confidence_interval_
            }
            
            print(f"✓ Overall median survival time: {self.kmf.median_survival_time_:.2f} months")
        else:
            # Survival curves by group
            groups = df[group_col].unique()
            
            for group in groups:
                mask = df[group_col] == group
                group_df = df[mask]
                
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(
                    durations=group_df[self.time_column],
                    event_observed=group_df[self.event_column],
                    label=str(group)
                )
                
                results[str(group)] = {
                    'survival_function': kmf_group.survival_function_,
                    'median_survival': kmf_group.median_survival_time_,
                    'confidence_interval': kmf_group.confidence_interval_,
                    'fitter': kmf_group
                }
                
                print(f"✓ {group}: median survival = {kmf_group.median_survival_time_:.2f} months")
        
        return results
    
    def plot_kaplan_meier(self, df: pd.DataFrame,
                         group_col: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot Kaplan-Meier survival curves.
        
        Args:
            df: DataFrame with time and event columns
            group_col: Optional column for grouping
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if group_col is None:
            # Overall curve
            self.kmf.fit(
                durations=df[self.time_column],
                event_observed=df[self.event_column],
                label='All Customers'
            )
            self.kmf.plot_survival_function(ax=ax, ci_show=True)
        else:
            # Curves by group
            groups = df[group_col].unique()
            colors = sns.color_palette("husl", len(groups))
            
            for idx, group in enumerate(groups):
                mask = df[group_col] == group
                group_df = df[mask]
                
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(
                    durations=group_df[self.time_column],
                    event_observed=group_df[self.event_column],
                    label=f'{group} (n={len(group_df)})'
                )
                kmf_group.plot_survival_function(ax=ax, ci_show=True, color=colors[idx])
        
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'Kaplan-Meier Survival Curves{" by " + group_col if group_col else ""}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Survival curve saved to {full_path}")
        
        return fig
    
    def fit_cox_model(self, df: pd.DataFrame,
                     feature_cols: List[str]) -> CoxPHFitter:
        """Fit Cox Proportional Hazards model.
        
        Args:
            df: DataFrame with time, event, and feature columns
            feature_cols: List of feature column names
            
        Returns:
            Fitted CoxPHFitter
        """
        print("\n" + "=" * 60)
        print("Cox Proportional Hazards Model")
        print("=" * 60)
        
        # Prepare data
        cols_to_use = [self.time_column, self.event_column] + feature_cols
        cox_df = df[cols_to_use].copy()
        
        # Fit model
        print(f"Fitting Cox model with {len(feature_cols)} features...")
        self.coxph.fit(
            cox_df,
            duration_col=self.time_column,
            event_col=self.event_column,
            show_progress=False
        )
        
        print("✓ Cox model fitted successfully")
        print(f"\nConcordance Index: {self.coxph.concordance_index_:.4f}")
        
        return self.coxph
    
    def get_cox_summary(self) -> pd.DataFrame:
        """Get summary of Cox model coefficients.
        
        Returns:
            DataFrame with hazard ratios and p-values
        """
        summary = self.coxph.summary.copy()
        summary = summary.sort_values('exp(coef)', ascending=False)
        
        print("\n" + "=" * 60)
        print("Cox Model - Top Risk Factors (Hazard Ratios)")
        print("=" * 60)
        print(summary[['exp(coef)', 'se(coef)', 'p']].head(10))
        
        return summary
    
    def plot_cox_coefficients(self, top_n: int = 15,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot Cox model coefficients (hazard ratios).
        
        Args:
            top_n: Number of top features to plot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get summary sorted by absolute coefficient
        summary = self.coxph.summary.copy()
        summary['abs_coef'] = summary['coef'].abs()
        summary = summary.sort_values('abs_coef', ascending=True).tail(top_n)
        
        # Plot hazard ratios
        colors = ['red' if x > 0 else 'green' for x in summary['coef']]
        ax.barh(range(len(summary)), summary['exp(coef)'] - 1, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(summary.index)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Hazard Ratio - 1', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Hazard Ratio (Cox Model)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Increases Churn Risk'),
            Patch(facecolor='green', alpha=0.7, label='Decreases Churn Risk')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            full_path = self.outputs_path / save_path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cox coefficients plot saved to {full_path}")
        
        return fig
    
    def predict_survival(self, X: pd.DataFrame,
                        times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Predict survival probabilities.
        
        Args:
            X: Features for prediction
            times: Time points for prediction (default: [3, 6, 12, 24, 36] months)
            
        Returns:
            DataFrame with survival probabilities
        """
        if times is None:
            times = np.array([3, 6, 12, 24, 36])
        
        survival_probs = self.coxph.predict_survival_function(X, times=times)
        
        return survival_probs
    
    def save_model(self, name: str = 'cox_model'):
        """Save Cox model to disk.
        
        Args:
            name: Model name for filename
        """
        outputs_path = Path(self.config.outputs.get('models_path', 'outputs/models/'))
        outputs_path.mkdir(parents=True, exist_ok=True)
        
        model_path = outputs_path / f"{name}.pkl"
        joblib.dump(self.coxph, model_path)
        
        print(f"✓ Cox model saved to {model_path}")


def main():
    """Main function to demonstrate survival analysis."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Survival Analysis for Customer Churn")
    print("=" * 60)
    
    # Load and prepare data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    # Clean data
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Initialize survival analysis
    survival = SurvivalAnalysis()
    
    # Kaplan-Meier by contract type
    print("\n1. Kaplan-Meier Analysis by Contract Type")
    km_results = survival.fit_kaplan_meier(df_features, group_col='Contract')
    survival.plot_kaplan_meier(df_features, group_col='Contract',
                               save_path='km_curve_by_contract.png')
    
    # Kaplan-Meier by tenure bucket
    print("\n2. Kaplan-Meier Analysis by Tenure Bucket")
    survival.plot_kaplan_meier(df_features, group_col='TenureBucket',
                               save_path='km_curve_by_tenure.png')
    
    # Cox Proportional Hazards
    print("\n3. Cox Proportional Hazards Model")
    
    # Select features for Cox model (numerical and some categorical)
    feature_cols = [
        'MonthlyCharges', 'ServiceCount', 'EngagementScore',
        'SeniorCitizen', 'ContractRisk'
    ]
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    cox_model = survival.fit_cox_model(df_features, feature_cols)
    summary = survival.get_cox_summary()
    survival.plot_cox_coefficients(save_path='cox_hazard_ratios.png')
    
    # Save model
    survival.save_model()
    
    print("\n✓ Survival analysis complete!")
    
    return survival


if __name__ == "__main__":
    main()
