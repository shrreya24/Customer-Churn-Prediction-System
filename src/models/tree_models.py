"""Tree-based models for churn prediction."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from typing import Optional, Dict, Tuple
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config
from src.utils.evaluation import ModelEvaluator


class TreeBasedModel:
    """Tree-based models wrapper (Random Forest and XGBoost)."""
    
    def __init__(self, model_type: str = 'random_forest',
                 config: Optional[Config] = None,
                 use_smote: bool = False):
        """Initialize tree-based model.
        
        Args:
            model_type: Type of model ('random_forest' or 'xgboost')
            config: Configuration object
            use_smote: Whether to use SMOTE for handling imbalanced data
        """
        if config is None:
            config = Config()
        self.config = config
        self.model_type = model_type
        self.use_smote = use_smote
        self.model_config = config.models.get(model_type, {})
        self.outputs_path = Path(config.outputs.get('models_path', 'outputs/models/'))
        self.model = None
        self.feature_names = None
        self.performance_metrics = {}
        self.smote = SMOTE(random_state=42) if use_smote else None
    
    def build_model(self):
        """Build model based on type.
        
        Returns:
            Scikit-learn compatible model
        """
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(**self.model_config)
        elif self.model_type == 'xgboost':
            model = XGBClassifier(**self.model_config, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> 'TreeBasedModel':
        """Train the tree-based model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping
            y_val: Optional validation target
            
        Returns:
            Self for method chaining
        """
        print("=" * 60)
        print(f"Training {self.model_type.replace('_', ' ').title()}")
        print("=" * 60)
        
        self.model = self.build_model()
        self.feature_names = X_train.columns.tolist()
        
        # Apply SMOTE if configured
        if self.use_smote:
            print(f"Original class distribution: {y_train.value_counts().to_dict()}")
            X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {y_train_resampled.value_counts().to_dict()}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Train model
        print(f"\nTraining on {len(X_train_resampled)} samples "
              f"with {len(self.feature_names)} features...")
        
        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            # Use early stopping for XGBoost
            self.model.fit(
                X_train_resampled, y_train_resampled,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_resampled, y_train_resampled)
        
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
        """Get feature importance.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, name: Optional[str] = None):
        """Save trained model to disk.
        
        Args:
            name: Model name for filename (defaults to model_type)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if name is None:
            name = self.model_type
        
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = self.outputs_path / f"{name}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save feature names and metrics
        metadata = {
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'model_config': self.model_config,
            'model_type': self.model_type,
            'use_smote': self.use_smote
        }
        
        metadata_path = self.outputs_path / f"{name}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    def load_model(self, name: Optional[str] = None):
        """Load trained model from disk.
        
        Args:
            name: Model name for filename (defaults to model_type)
        """
        if name is None:
            name = self.model_type
        
        model_path = self.outputs_path / f"{name}.pkl"
        metadata_path = self.outputs_path / f"{name}_metadata.pkl"
        
        self.model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        
        self.feature_names = metadata['feature_names']
        self.performance_metrics = metadata['performance_metrics']
        self.model_type = metadata.get('model_type', self.model_type)
        self.use_smote = metadata.get('use_smote', False)
        
        print(f"✓ Model loaded from {model_path}")


def train_all_models(X_train, X_test, y_train, y_test) -> Dict[str, TreeBasedModel]:
    """Train all tree-based models and compare.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dictionary of model name to trained model
    """
    models = {}
    
    # Random Forest
    print("\n" + "=" * 60)
    print("1. Random Forest")
    print("=" * 60)
    rf = TreeBasedModel(model_type='random_forest')
    rf.train(X_train, y_train)
    rf.evaluate(X_test, y_test)
    rf.save_model('random_forest')
    models['Random Forest'] = rf
    
    # XGBoost
    print("\n" + "=" * 60)
    print("2. XGBoost")
    print("=" * 60)
    xgb = TreeBasedModel(model_type='xgboost')
    xgb.train(X_train, y_train)
    xgb.evaluate(X_test, y_test)
    xgb.save_model('xgboost')
    models['XGBoost'] = xgb
    
    # XGBoost with SMOTE
    print("\n" + "=" * 60)
    print("3. XGBoost with SMOTE")
    print("=" * 60)
    xgb_smote = TreeBasedModel(model_type='xgboost', use_smote=True)
    xgb_smote.train(X_train, y_train)
    xgb_smote.evaluate(X_test, y_test)
    xgb_smote.save_model('xgboost_smote')
    models['XGBoost + SMOTE'] = xgb_smote
    
    return models


def main():
    """Main function to train and evaluate tree-based models."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Tree-Based Models Training")
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
    
    # Note: Tree models don't require scaling, but we can still use scaled features
    # For consistency, let's scale
    scale_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                      'AvgMonthlySpend', 'ServiceCount', 'EngagementScore']
    X_train = preprocessor.scale_features(X_train, scale_features, fit=True)
    X_test = preprocessor.scale_features(X_test, scale_features, fit=False)
    
    # Train all models
    models = train_all_models(X_train, X_test, y_train, y_test)
    
    # Compare feature importance
    print("\n" + "=" * 60)
    print("Feature Importance Comparison")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n{name} - Top 10 Features:")
        importance = model.get_feature_importance()
        print(importance.head(10).to_string(index=False))
    
    return models


if __name__ == "__main__":
    main()
