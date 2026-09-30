"""Streamlit web application for customer churn prediction and analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.tree_models import TreeBasedModel
from src.explainability.shap_analysis import SHAPAnalyzer
from src.utils.config import Config

# Page configuration
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .churn-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .churn-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and preprocessors."""
    config = Config()
    models_path = Path(config.outputs.get('models_path', 'outputs/models/'))
    
    # Load XGBoost model (best performer)
    model_path = models_path / 'xgboost.pkl'
    metadata_path = models_path / 'xgboost_metadata.pkl'
    
    if not model_path.exists():
        return None, None, None, None
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    try:
        preprocessor.load_processed_data()
    except:
        pass
    
    # Load feature engineer
    engineer = FeatureEngineer()
    
    return model, metadata, preprocessor, engineer


def prepare_input_features(user_input, preprocessor, engineer):
    """Prepare user input for prediction."""
    # Create DataFrame from user input
    df_input = pd.DataFrame([user_input])
    
    # Engineer features
    df_features = engineer.create_all_features(df_input)
    
    # Encode features
    df_encoded = preprocessor.encode_features(df_features, fit=False)
    
    # Remove Churn and customerID if present
    exclude_cols = ['Churn', 'customerID']
    df_encoded = df_encoded.drop(columns=[col for col in exclude_cols if col in df_encoded.columns])
    
    return df_encoded


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Retention Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Churn Prediction & Explainability System")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Telco+Analytics", use_container_width=True)
        st.markdown("---")
        page = st.radio("Navigation", ["🎯 Churn Prediction", "📈 Model Insights", "ℹ️ About"])
        st.markdown("---")
        st.markdown("### System Info")
        st.markdown("**Model:** XGBoost")
        st.markdown("**Dataset:** IBM Telco Churn")
        st.markdown("**Features:** 20+")
    
    # Load models
    model, metadata, preprocessor, engineer = load_models()
    
    if model is None:
        st.error("⚠️ Models not found. Please train models first by running the training scripts.")
        st.info("Run: `python src/models/tree_models.py`")
        return
    
    # Page routing
    if page == "🎯 Churn Prediction":
        churn_prediction_page(model, metadata, preprocessor, engineer)
    elif page == "📈 Model Insights":
        model_insights_page(metadata)
    else:
        about_page()


def churn_prediction_page(model, metadata, preprocessor, engineer):
    """Churn prediction page."""
    st.markdown('<h2 class="sub-header">🎯 Single Customer Churn Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("Enter key customer information to predict churn risk.")
    st.info("💡 **Tip:** Only 8 essential fields needed - we'll use smart defaults for the rest!")
    
    # Create simplified input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Customer Profile")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer 65+ years old?")
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with us?")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 5.0, 
                                         help="Current monthly bill amount")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], 
                                       help="Type of internet service")
    
    with col2:
        st.markdown("#### 💳 Contract & Support")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                               help="Current contract commitment level")
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], 
                                   help="Does customer have tech support?")
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"],
                                     help="How does the customer pay?")
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Prepare input with smart defaults
        total_charges = monthly_charges * max(tenure, 1)
        
        # Set intelligent defaults based on internet service
        if internet_service == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support_val = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
        else:
            online_security = "Yes" if tech_support == "Yes" else "No"
            online_backup = "Yes" if contract != "Month-to-month" else "No"
            device_protection = "Yes" if contract != "Month-to-month" else "No"
            tech_support_val = tech_support
            streaming_tv = "Yes" if monthly_charges > 70 else "No"
            streaming_movies = "Yes" if monthly_charges > 70 else "No"
        
        user_input = {
            'gender': "Male",  # Default - least predictive feature
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': "Yes",  # Default
            'Dependents': "No",  # Default
            'tenure': tenure,
            'PhoneService': "Yes",  # Most customers have phone
            'MultipleLines': "No",  # Default
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support_val,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': "Yes",  # Default - most common
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
        }
        
        try:
            # Prepare features
            X_input = prepare_input_features(user_input, preprocessor, engineer)
            
            # Align features with model
            model_features = metadata['feature_names']
            
            # Add missing features with 0
            for feat in model_features:
                if feat not in X_input.columns:
                    X_input[feat] = 0
            
            # Reorder to match model
            X_input = X_input[model_features]
            
            # Predict
            churn_prob = model.predict_proba(X_input)[0, 1]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            # Churn probability
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Gauge chart
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh([0], [churn_prob], color='#d32f2f' if churn_prob > 0.5 else '#388e3c', height=0.5)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_yticks([])
                ax.set_xlabel('Churn Probability', fontsize=12, fontweight='bold')
                ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.metric(
                    label="Churn Risk",
                    value=f"{churn_prob:.1%}",
                    delta=f"{(churn_prob - 0.5) * 100:.1f}% vs avg" if churn_prob != 0.5 else None
                )
            
            with col3:
                risk_level = "🔴 HIGH" if churn_prob > 0.7 else "🟡 MEDIUM" if churn_prob > 0.4 else "🟢 LOW"
                st.metric(label="Risk Level", value=risk_level)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if churn_prob > 0.6:
                st.warning("⚠️ **High Churn Risk - Immediate Action Required**")
                recommendations = []
                
                if contract == "Month-to-month":
                    recommendations.append("✅ Offer long-term contract incentives (1-year or 2-year discount)")
                if tech_support in ["No", "No internet service"]:
                    recommendations.append("✅ Provide complimentary tech support for 3 months")
                if tenure < 12:
                    recommendations.append("✅ Implement early engagement program for new customers")
                if payment_method == "Electronic check":
                    recommendations.append("✅ Encourage automatic payment methods with small discount")
                if monthly_charges > 70:
                    recommendations.append("✅ Review pricing and consider loyalty discount")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            elif churn_prob > 0.3:
                st.info("ℹ️ **Medium Churn Risk - Proactive Engagement Recommended**")
                st.markdown("- 📞 Schedule check-in call to assess satisfaction")
                st.markdown("- 🎁 Offer service upgrade or bundle discount")
                st.markdown("- 📧 Enroll in loyalty program")
            
            else:
                st.success("✅ **Low Churn Risk - Customer is Stable**")
                st.markdown("- 🌟 Continue current engagement strategy")
                st.markdown("- 💬 Request feedback for testimonials")
                st.markdown("- 🔄 Monitor for any service changes")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")


def model_insights_page(metadata):
    """Model insights page."""
    st.markdown('<h2 class="sub-header">📈 Model Performance & Insights</h2>', unsafe_allow_html=True)
    
    if metadata is None or 'performance_metrics' not in metadata:
        st.warning("Model metrics not available. Please train models first.")
        return
    
    metrics = metadata['performance_metrics']
    
    # Performance metrics
    st.markdown("### 🎯 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    
    # Feature importance (if available)
    st.markdown("### 🔍 Top Churn Drivers")
    
    st.markdown("""
    Based on analysis of thousands of customers, the key factors influencing churn are:
    
    1. **Contract Type** - Month-to-month contracts have 3x higher churn
    2. **Tenure** - New customers (< 12 months) are most at risk
    3. **Tech Support** - Customers without tech support churn more frequently
    4. **Monthly Charges** - High charges correlate with increased churn
    5. **Payment Method** - Electronic check users show higher churn rates
    6. **Internet Service** - Fiber optic users have higher churn (possibly due to pricing)
    7. **Service Count** - Fewer subscribed services = higher churn risk
    8. **Senior Citizens** - Higher churn rates in senior demographic
    """)
    
    # Model config
    with st.expander("⚙️ Model Configuration"):
        st.json(metadata.get('model_config', {}))


def about_page():
    """About page."""
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Customer Retention Analytics System
    
    ### 🎯 Purpose
    This end-to-end machine learning system predicts customer churn and provides actionable insights
    to improve retention. Unlike simple classification models, this system focuses on:
    
    - **Understanding WHY customers churn**
    - **Identifying key intervention points**
    - **Providing explainable predictions**
    - **Estimating time-to-churn**
    - **Causal analysis of retention strategies**
    
    ### 📊 Dataset
    **IBM Telco Customer Churn Dataset** (~7,000 customers)
    - Customer demographics (gender, age, dependents)
    - Service subscriptions (phone, internet, streaming)
    - Contract details (type, tenure, billing)
    - Churn label (Yes/No)
    
    ### 🤖 Models
    
    #### Baseline
    - **Logistic Regression** - Interpretable linear model
    
    #### Tree-Based
    - **Random Forest** - Ensemble learning with feature interactions
    - **XGBoost** - Gradient boosting for optimal performance
    
    #### Advanced Analytics
    - **Survival Analysis** - Kaplan-Meier curves and Cox Proportional Hazards
    - **Uplift Modeling** - S-learner and T-learner for intervention analysis
    
    ### 🔍 Explainability
    - **SHAP Values** - Understand individual prediction drivers
    - **Feature Importance** - Global view of churn factors
    - **Partial Dependence** - Feature effect visualization
    
    ### 💡 Key Features Analyzed
    - Average Monthly Spend
    - Service Count
    - Tenure Buckets
    - Engagement Score
    - Contract Risk Score
    - Payment Method Patterns
    - Support Availability
    
    ### 📈 Business Impact
    This system enables:
    - ✅ Proactive customer retention strategies
    - ✅ Targeted intervention campaigns
    - ✅ Resource optimization for retention teams
    - ✅ Data-driven decision making
    - ✅ Reduced customer acquisition costs
    
    ### 🔬 Technology Stack
    - **ML:** scikit-learn, XGBoost, LightGBM
    - **Survival:** lifelines
    - **Causal:** econml
    - **Explainability:** SHAP, LIME
    - **UI:** Streamlit
    - **Visualization:** Matplotlib, Seaborn, Plotly
    
    ---
    
    **Developed by:** ML Research Team  
    **Last Updated:** January 2026  
    **Version:** 1.0.0
    """)


if __name__ == "__main__":
    main()
