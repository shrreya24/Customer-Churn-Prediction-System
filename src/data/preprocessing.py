"""Data preprocessing module for cleaning and encoding."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, List
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class DataPreprocessor:
    """Handle data cleaning, encoding, and splitting."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.processed_path = Path(config.data['processed_path'])
        self.exclude_features = config.features.get('exclude_features', [])
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (handle spaces)
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(
                df_clean['TotalCharges'], errors='coerce'
            )
            
            # Fill missing TotalCharges with 0 for customers with tenure = 0
            mask = (df_clean['TotalCharges'].isnull()) & (df_clean['tenure'] == 0)
            df_clean.loc[mask, 'TotalCharges'] = 0
            
            # For others, fill with median
            if df_clean['TotalCharges'].isnull().any():
                median_charges = df_clean['TotalCharges'].median()
                df_clean['TotalCharges'].fillna(median_charges, inplace=True)
        
        # Convert Churn to binary
        if df_clean['Churn'].dtype == 'object':
            df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)
        
        # Convert SeniorCitizen to int if exists
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)
        
        # Handle 'No internet service' and 'No phone service'
        # Replace with 'No' for consistency
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].replace({
                    'No internet service': 'No',
                    'No phone service': 'No'
                })
        
        print(f"✓ Data cleaned. Shape: {df_clean.shape}")
        print(f"  Missing values: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit encoders (True for train, False for test)
            
        Returns:
            DataFrame with encoded features
        """
        print("Encoding categorical features...")
        df_encoded = df.copy()
        
        # Identify categorical columns (excluding target and IDs)
        categorical_cols = df_encoded.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
        
        # Remove excluded features
        categorical_cols = [col for col in categorical_cols 
                           if col not in self.exclude_features]
        
        # Binary encoding for Yes/No columns
        binary_cols = []
        for col in categorical_cols:
            unique_vals = df_encoded[col].unique()
            if set(unique_vals).issubset({'Yes', 'No'}):
                df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
                binary_cols.append(col)
        
        # Remove binary columns from categorical list
        categorical_cols = [col for col in categorical_cols if col not in binary_cols]
        
        # One-hot encoding for remaining categorical features
        if categorical_cols:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=categorical_cols,
                prefix=categorical_cols,
                drop_first=True  # Avoid multicollinearity
            )
        
        print(f"✓ Features encoded. Binary: {len(binary_cols)}, "
              f"One-hot: {len(categorical_cols)}")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, features: List[str],
                       fit: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: DataFrame with features to scale
            features: List of feature names to scale
            fit: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            DataFrame with scaled features
        """
        print("Scaling numerical features...")
        df_scaled = df.copy()
        
        # Filter features that exist in DataFrame
        features_to_scale = [f for f in features if f in df_scaled.columns]
        
        if features_to_scale:
            if fit:
                df_scaled[features_to_scale] = self.scaler.fit_transform(
                    df_scaled[features_to_scale]
                )
            else:
                df_scaled[features_to_scale] = self.scaler.transform(
                    df_scaled[features_to_scale]
                )
            
            print(f"✓ Scaled {len(features_to_scale)} numerical features")
        
        return df_scaled
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target, remove excluded columns.
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Get target
        y = df['Churn'].copy()
        
        # Get features (drop target and excluded features)
        cols_to_drop = ['Churn'] + self.exclude_features
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        X = df.drop(columns=cols_to_drop)
        
        print(f"✓ Prepared features: {X.shape[1]} features, {len(y)} samples")
        print(f"  Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: Optional[float] = None,
                   random_state: Optional[int] = None,
                   stratify: Optional[bool] = None) -> Tuple:
        """Split data into train and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.data['test_size']
        if random_state is None:
            random_state = self.config.data['random_state']
        if stratify is None:
            stratify = self.config.data['stratify']
        
        print(f"Splitting data (test_size={test_size}, stratify={stratify})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"  - Churn rate: {y_train.mean():.2%}")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        print(f"  - Churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                           scale_features: Optional[List[str]] = None) -> Tuple:
        """Run complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            scale_features: List of features to scale
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("Running Preprocessing Pipeline")
        print("=" * 60)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_clean, fit=True)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale numerical features if specified
        if scale_features:
            X_train = self.scale_features(X_train, scale_features, fit=True)
            X_test = self.scale_features(X_test, scale_features, fit=False)
        
        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        print("\n✓ Preprocessing pipeline complete!")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series):
        """Save processed data to disk.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        """
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        X_train.to_csv(self.processed_path / "X_train.csv", index=False)
        X_test.to_csv(self.processed_path / "X_test.csv", index=False)
        y_train.to_csv(self.processed_path / "y_train.csv", index=False)
        y_test.to_csv(self.processed_path / "y_test.csv", index=False)
        
        # Save scaler
        joblib.dump(self.scaler, self.processed_path / "scaler.pkl")
        
        print(f"✓ Processed data saved to {self.processed_path}")
    
    def load_processed_data(self) -> Tuple:
        """Load processed data from disk.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Loading processed data from {self.processed_path}")
        
        X_train = pd.read_csv(self.processed_path / "X_train.csv")
        X_test = pd.read_csv(self.processed_path / "X_test.csv")
        y_train = pd.read_csv(self.processed_path / "y_train.csv").squeeze()
        y_test = pd.read_csv(self.processed_path / "y_test.csv").squeeze()
        
        # Load scaler
        scaler_path = self.processed_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        print(f"✓ Loaded: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test


def main():
    """Main function to demonstrate preprocessing."""
    from src.data.ingestion import DataIngestion
    
    print("=" * 60)
    print("Customer Retention - Data Preprocessing")
    print("=" * 60)
    
    # Load data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    # Features to scale
    scale_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, scale_features=scale_features
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Features: {X_train.columns.tolist()[:10]}... ({len(X_train.columns)} total)")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
