"""Baseline models for churn prediction."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Tuple
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config
from src.utils.evaluation import ModelEvaluator


class BaselineModel:
    """Logistic Regression baseline model."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize baseline model.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = Config()
        self.config = config
        self.model_config = config.models.get('logistic_regression', {})
        self.outputs_path = Path(config.outputs.get('models_path', 'outputs/models/'))
        self.model = None
        self.feature_names = None
        self.performance_metrics = {}
    
    def build_model(self) -> LogisticRegression:
        """Build logistic regression model with configured parameters.
        
        Returns:
            LogisticRegression model
        """
        model = LogisticRegression(**self.model_config)
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaselineModel':
        """Train the baseline model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Self for method chaining
        """
        print("=" * 60)
        print("Training Logistic Regression (Baseline Model)")
        print("=" * 60)
        
        self.model = self.build_model()
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        print(f"Training on {len(X_train)} samples with {len(self.feature_names)} features...")
        self.model.fit(X_train, y_train)
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        print(f"✓ Training accuracy: {train_score:.4f}")
        
        # Cross-validation
        cv_folds = self.config.evaluation.get('cv_folds', 5)
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )
        
        print(f"✓ CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.performance_metrics['cv_roc_auc_mean'] = cv_scores.mean()
        self.performance_metrics['cv_roc_auc_std'] = cv_scores.std()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities for class 1 (churn)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating on Test Set")
        print("=" * 60)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        
        # Create evaluator
        evaluator = ModelEvaluator(y_test, y_pred_proba)
        metrics = evaluator.compute_metrics()
        
        # Print metrics
        print("\nTest Set Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store metrics
        self.performance_metrics.update(metrics)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (coefficients).
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def save_model(self, name: str = 'logistic_regression'):
        """Save trained model to disk.
        
        Args:
            name: Model name for filename
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = self.outputs_path / f"{name}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save feature names and metrics
        metadata = {
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'model_config': self.model_config
        }
        
        metadata_path = self.outputs_path / f"{name}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    def load_model(self, name: str = 'logistic_regression'):
        """Load trained model from disk.
        
        Args:
            name: Model name for filename
        """
        model_path = self.outputs_path / f"{name}.pkl"
        metadata_path = self.outputs_path / f"{name}_metadata.pkl"
        
        self.model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        
        self.feature_names = metadata['feature_names']
        self.performance_metrics = metadata['performance_metrics']
        
        print(f"✓ Model loaded from {model_path}")


def main():
    """Main function to train and evaluate baseline model."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Baseline Model Training")
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
    
    # Encode and prepare
    df_encoded = preprocessor.encode_features(df_features, fit=True)
    X, y = preprocessor.prepare_features_target(df_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale features
    scale_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                      'AvgMonthlySpend', 'ServiceCount', 'EngagementScore']
    X_train = preprocessor.scale_features(X_train, scale_features, fit=True)
    X_test = preprocessor.scale_features(X_test, scale_features, fit=False)
    
    # Train baseline model
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    
    # Evaluate
    metrics = baseline.evaluate(X_test, y_test)
    
    # Feature importance
    importance = baseline.get_feature_importance()
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features")
    print("=" * 60)
    print(importance.head(10).to_string(index=False))
    
    # Save model
    baseline.save_model()
    
    return baseline, metrics


if __name__ == "__main__":
    main()
