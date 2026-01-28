"""Feature engineering module for creating derived features."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class FeatureEngineer:
    """Create engineered features for churn prediction."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize feature engineer.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.tenure_buckets = config.features.get('tenure_buckets', [])
        self.service_features = config.features.get('service_features', [])
        self.features_path = Path(config.features.get('features_path', 'data/features/'))
    
    def create_average_monthly_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create average monthly spend feature.
        
        Args:
            df: DataFrame with tenure and TotalCharges
            
        Returns:
            DataFrame with new feature
        """
        df = df.copy()
        
        # Avoid division by zero
        df['AvgMonthlySpend'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']  # Use current charges for tenure=0
        )
        
        return df
    
    def create_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count total number of services subscribed.
        
        Args:
            df: DataFrame with service columns
            
        Returns:
            DataFrame with service count
        """
        df = df.copy()
        
        service_cols = [col for col in self.service_features if col in df.columns]
        
        # Count services (assuming 'Yes' means subscribed)
        df['ServiceCount'] = 0
        for col in service_cols:
            if df[col].dtype == 'object':
                # If not yet encoded, count 'Yes' values
                df['ServiceCount'] += (df[col].isin(['Yes', 'DSL', 'Fiber optic'])).astype(int)
            else:
                # If already encoded as binary
                df['ServiceCount'] += df[col]
        
        return df
    
    def create_tenure_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical tenure buckets.
        
        Args:
            df: DataFrame with tenure column
            
        Returns:
            DataFrame with tenure bucket feature
        """
        df = df.copy()
        
        # Create buckets based on configuration
        bins = [bucket[0] for bucket in self.tenure_buckets] + [float('inf')]
        labels = [bucket[2] for bucket in self.tenure_buckets]
        
        df['TenureBucket'] = pd.cut(
            df['tenure'],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )
        
        return df
    
    def create_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement proxy score.
        
        Combines tenure and service count to estimate customer engagement.
        
        Args:
            df: DataFrame with relevant features
            
        Returns:
            DataFrame with engagement score
        """
        df = df.copy()
        
        # Ensure ServiceCount exists
        if 'ServiceCount' not in df.columns:
            df = self.create_service_count(df)
        
        # Normalize tenure (0-1 scale)
        tenure_normalized = df['tenure'] / df['tenure'].max()
        
        # Normalize service count (0-1 scale)
        service_normalized = df['ServiceCount'] / df['ServiceCount'].max()
        
        # Weighted combination (tenure weighted more heavily)
        df['EngagementScore'] = (0.6 * tenure_normalized + 0.4 * service_normalized)
        
        return df
    
    def create_contract_payment_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between contract and payment method.
        
        Args:
            df: DataFrame with Contract and PaymentMethod columns
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
            # Create combined feature
            df['Contract_Payment'] = (
                df['Contract'].astype(str) + '_' + df['PaymentMethod'].astype(str)
            )
        
        return df
    
    def create_support_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary flag for tech support availability.
        
        Args:
            df: DataFrame with TechSupport column
            
        Returns:
            DataFrame with support flag
        """
        df = df.copy()
        
        if 'TechSupport' in df.columns:
            if df['TechSupport'].dtype == 'object':
                df['HasTechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
            else:
                # Already binary
                df['HasTechSupport'] = df['TechSupport']
        
        return df
    
    def create_billing_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create flag for paperless billing.
        
        Args:
            df: DataFrame with PaperlessBilling column
            
        Returns:
            DataFrame with billing flag
        """
        df = df.copy()
        
        if 'PaperlessBilling' in df.columns:
            if df['PaperlessBilling'].dtype == 'object':
                df['IsPaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
            else:
                df['IsPaperlessBilling'] = df['PaperlessBilling']
        
        return df
    
    def create_contract_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk score based on contract type.
        
        Month-to-month contracts have highest churn risk.
        
        Args:
            df: DataFrame with Contract column
            
        Returns:
            DataFrame with risk score
        """
        df = df.copy()
        
        if 'Contract' in df.columns:
            # Risk mapping
            risk_map = {
                'Month-to-month': 3,
                'One year': 2,
                'Two year': 1
            }
            
            df['ContractRisk'] = df['Contract'].map(risk_map).fillna(2)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features.
        
        Args:
            df: Raw/cleaned DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating engineered features...")
        df_features = df.copy()
        
        # Create all features
        df_features = self.create_average_monthly_spend(df_features)
        df_features = self.create_service_count(df_features)
        df_features = self.create_tenure_buckets(df_features)
        df_features = self.create_engagement_score(df_features)
        df_features = self.create_contract_payment_interactions(df_features)
        df_features = self.create_support_flag(df_features)
        df_features = self.create_billing_flag(df_features)
        df_features = self.create_contract_risk_score(df_features)
        
        # Count new features
        original_cols = set(df.columns)
        new_cols = set(df_features.columns) - original_cols
        
        print(f"✓ Created {len(new_cols)} engineered features:")
        for col in sorted(new_cols):
            print(f"  - {col}")
        
        return df_features
    
    def save_feature_descriptions(self):
        """Save descriptions of engineered features."""
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        descriptions = {
            'AvgMonthlySpend': 'Total charges divided by tenure (average spend per month)',
            'ServiceCount': 'Total number of services subscribed (phone, internet, security, etc.)',
            'TenureBucket': 'Categorical grouping of customer tenure',
            'EngagementScore': 'Composite score from tenure and service count (0-1)',
            'Contract_Payment': 'Interaction between contract type and payment method',
            'HasTechSupport': 'Binary flag for tech support availability',
            'IsPaperlessBilling': 'Binary flag for paperless billing',
            'ContractRisk': 'Risk score based on contract type (3=month-to-month, 1=two year)',
        }
        
        desc_df = pd.DataFrame([
            {'Feature': k, 'Description': v}
            for k, v in descriptions.items()
        ])
        
        desc_path = self.features_path / 'feature_descriptions.csv'
        desc_df.to_csv(desc_path, index=False)
        
        print(f"✓ Feature descriptions saved to {desc_path}")


def main():
    """Main function to demonstrate feature engineering."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    
    print("=" * 60)
    print("Customer Retention - Feature Engineering")
    print("=" * 60)
    
    # Load data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    # Clean data first
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Save feature descriptions
    engineer.save_feature_descriptions()
    
    print("\n" + "=" * 60)
    print("Feature Engineering Summary")
    print("=" * 60)
    print(f"Total columns: {len(df_features.columns)}")
    print(f"\nSample of new features:")
    new_feature_cols = ['AvgMonthlySpend', 'ServiceCount', 'EngagementScore',
                        'TenureBucket', 'ContractRisk']
    existing_cols = [col for col in new_feature_cols if col in df_features.columns]
    print(df_features[existing_cols].head())
    
    return df_features


if __name__ == "__main__":
    main()
