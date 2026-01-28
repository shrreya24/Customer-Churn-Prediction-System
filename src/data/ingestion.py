"""Data ingestion module for downloading and loading the Telco churn dataset."""
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class DataIngestion:
    """Handle data downloading and loading from Kaggle."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize data ingestion.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.raw_path = Path(config.data['raw_path'])
        self.dataset_name = config.data['kaggle_dataset']
        
    def download_from_kaggle(self, force: bool = False) -> Path:
        """Download dataset from Kaggle.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded dataset
        """
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        # Expected file path
        file_path = self.raw_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        if file_path.exists() and not force:
            print(f"Dataset already exists at {file_path}")
            return file_path
        
        print(f"Downloading dataset from Kaggle: {self.dataset_name}")
        
        try:
            # Import kaggle library
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Authenticate
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                self.dataset_name,
                path=str(self.raw_path),
                unzip=True
            )
            
            print(f"✓ Dataset downloaded successfully to {self.raw_path}")
            
            # Save metadata
            metadata = {
                'download_date': datetime.now().isoformat(),
                'dataset': self.dataset_name,
                'source': 'kaggle'
            }
            
            metadata_path = self.raw_path / "metadata.txt"
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            return file_path
            
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            print("\nPlease ensure:")
            print("1. You have a Kaggle account")
            print("2. API credentials are set up at ~/.kaggle/kaggle.json")
            print("3. You have accepted the dataset terms on Kaggle website")
            raise
    
    def load_data(self, download: bool = True) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame.
        
        Args:
            download: Whether to download if file doesn't exist
            
        Returns:
            DataFrame with the dataset
        """
        file_path = self.raw_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        if not file_path.exists():
            if download:
                file_path = self.download_from_kaggle()
            else:
                raise FileNotFoundError(
                    f"Dataset not found at {file_path}. "
                    f"Set download=True to download from Kaggle."
                )
        
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"✓ Loaded dataset with shape: {df.shape}")
        print(f"  Columns: {df.shape[1]}")
        print(f"  Rows: {df.shape[0]}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate the loaded dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for expected columns
        expected_columns = ['customerID', 'gender', 'tenure', 'MonthlyCharges',
                          'TotalCharges', 'Churn']
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing expected columns: {missing_cols}")
        
        # Check for minimum row count
        if len(df) < 1000:
            issues.append(f"Dataset too small: {len(df)} rows (expected > 1000)")
        
        # Check for churn column
        if 'Churn' in df.columns:
            if df['Churn'].dtype == 'object':
                unique_values = df['Churn'].unique()
                if not set(unique_values).issubset({'Yes', 'No'}):
                    issues.append(f"Unexpected churn values: {unique_values}")
        else:
            issues.append("Churn column not found")
        
        # Check for missing data
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            issues.append(f"Columns with >50% missing: {high_missing.to_dict()}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            print("✓ Data validation passed")
        else:
            print("⚠ Data validation found issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        return is_valid, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
        }
        
        if 'Churn' in df.columns:
            churn_dist = df['Churn'].value_counts().to_dict()
            summary['churn_distribution'] = churn_dist
            if df['Churn'].dtype == 'object':
                churn_rate = (df['Churn'] == 'Yes').mean()
            else:
                churn_rate = df['Churn'].mean()
            summary['churn_rate'] = f"{churn_rate:.2%}"
        
        return summary


def main():
    """Main function to demonstrate data ingestion."""
    print("=" * 60)
    print("Customer Retention - Data Ingestion")
    print("=" * 60)
    
    # Initialize ingestion
    ingestion = DataIngestion()
    
    # Download and load data
    df = ingestion.load_data(download=True)
    
    # Validate
    is_valid, issues = ingestion.validate_data(df)
    
    # Get summary
    summary = ingestion.get_data_summary(df)
    
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, dict) and len(value) > 5:
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    print("\n✓ Data ingestion complete!")
    
    return df


if __name__ == "__main__":
    main()
