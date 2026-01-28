#!/bin/bash

# Complete pipeline execution script for Customer Retention ML System

echo "======================================"
echo "Customer Retention ML System"
echo "Complete Pipeline Execution"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Step 1: Data Ingestion"
echo "======================================"
python src/data/ingestion.py

echo ""
echo "======================================"
echo "Step 2: Data Preprocessing"
echo "======================================"
python src/data/preprocessing.py

echo ""
echo "======================================"
echo "Step 3: Feature Engineering"
echo "======================================"
python src/data/feature_engineering.py

echo ""
echo "======================================"
echo "Step 4: Train Baseline Model"
echo "======================================"
python src/models/baseline.py

echo ""
echo "======================================"
echo "Step 5: Train Tree-Based Models"
echo "======================================"
python src/models/tree_models.py

echo ""
echo "======================================"
echo "Step 6: Survival Analysis"
echo "======================================"
python src/models/survival.py

echo ""
echo "======================================"
echo "Step 7: SHAP Explainability"
echo "======================================"
python src/explainability/shap_analysis.py

echo ""
echo "======================================"
echo "Step 8: Feature Importance Analysis"
echo "======================================"
python src/explainability/feature_importance.py

echo ""
echo "======================================"
echo "Step 9: Causal Inference & Uplift"
echo "======================================"
python src/causal/uplift_models.py

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  - Models: outputs/models/"
echo "  - Figures: outputs/figures/"
echo "  - Processed data: data/processed/"
echo ""
echo "To launch the Streamlit UI, run:"
echo "  streamlit run app/streamlit_app.py"
echo ""
