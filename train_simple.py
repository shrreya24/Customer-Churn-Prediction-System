"""Simple XGBoost training script using preprocessed data."""
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("="*60)
print("Training XGBoost Model")
print("="*60)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Test: {X_test.shape[0]} samples")

# Train XGBoost
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✓ Model trained!")

# Evaluate
print("\nEvaluating on test set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n{'='*60}")
print("Test Set Performance")
print(f"{'='*60}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Save model
Path("outputs/models").mkdir(parents=True, exist_ok=True)
model_path = "outputs/models/xgboost.pkl"
joblib.dump(model, model_path)

metadata = {
    'feature_names': X_train.columns.tolist(),
    'performance_metrics': {
        'roc_auc': float(roc_auc),
    },
    'model_type': 'xgboost'
}
joblib.dump(metadata, "outputs/models/xgboost_metadata.pkl")

print(f"\n✓ Model saved to {model_path}")
print(f"✓ Metadata saved")
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print("\nYou can now run the Streamlit UI:")
print("  streamlit run app/streamlit_app.py")
