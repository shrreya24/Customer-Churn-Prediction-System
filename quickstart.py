"""
Quick start script to train models and launch Streamlit UI.

This script provides a simplified entry point for running the system.
Usage:
    python quickstart.py          # Interactive mode
    python quickstart.py --auto   # Run complete pipeline automatically
    python quickstart.py --ui     # Launch Streamlit UI only
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in: {description}")
        return False
    
    print(f"✓ {description} completed")
    return True


def run_complete_pipeline(launch_ui=True):
    """Run the complete ML pipeline."""
    steps = [
        ("python src/data/ingestion.py", "Data Ingestion"),
        ("python src/data/preprocessing.py", "Data Preprocessing"),
        ("python src/data/feature_engineering.py", "Feature Engineering"),
        ("python src/models/baseline.py", "Baseline Model Training"),
        ("python src/models/tree_models.py", "Tree-Based Models Training"),
        ("python src/explainability/shap_analysis.py", "SHAP Analysis"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print("\n❌ Pipeline failed. Please check errors above.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ Pipeline completed successfully!")
    print("="*60)
    
    if launch_ui:
        print("\n🚀 Launching Streamlit UI...")
        subprocess.run("streamlit run app/streamlit_app.py", shell=True)


def main():
    """Main execution flow."""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Customer Retention ML System - Quick Start          ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ Error: config.yaml not found. Please run from project root.")
        sys.exit(1)
    
    # Handle command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--auto', '-a', '1']:
            print("🚀 Running complete pipeline automatically...")
            run_complete_pipeline(launch_ui=True)
            return
        elif arg in ['--ui', '-u', '3']:
            print("\n🚀 Launching Streamlit UI...")
            subprocess.run("streamlit run app/streamlit_app.py", shell=True)
            return
        elif arg in ['--train', '-t', '2']:
            steps = [
                ("python src/models/baseline.py", "Baseline Model"),
                ("python src/models/tree_models.py", "Tree-Based Models"),
            ]
            for cmd, desc in steps:
                run_command(cmd, desc)
            print("\n✓ Models trained successfully!")
            return
    
    print("\nChoose an option:")
    print("1. Run complete pipeline (data + training + analysis)")
    print("2. Train models only (requires preprocessed data)")
    print("3. Launch Streamlit UI only")
    print("4. Run quick demo (minimal training for testing)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_complete_pipeline(launch_ui=False)
        launch = input("\nLaunch Streamlit UI? (y/n): ").strip().lower()
        if launch == 'y':
            run_command("streamlit run app/streamlit_app.py", "Launching Streamlit")
    
    elif choice == "2":
        # Train models only
        steps = [
            ("python src/models/baseline.py", "Baseline Model"),
            ("python src/models/tree_models.py", "Tree-Based Models"),
        ]
        
        for cmd, desc in steps:
            run_command(cmd, desc)
        
        print("\n✓ Models trained successfully!")
    
    elif choice == "3":
        # Launch UI only
        print("\n🚀 Launching Streamlit UI...")
        subprocess.run("streamlit run app/streamlit_app.py", shell=True)
    
    elif choice == "4":
        # Quick demo
        print("\n🚀 Running quick demo...")
        print("(This will train XGBoost only with minimal data)")
        
        run_command("python src/data/ingestion.py", "Data Download")
        run_command("python src/data/preprocessing.py", "Data Preprocessing")
        run_command("python src/models/tree_models.py", "XGBoost Training")
        
        print("\n✓ Demo complete! Launching UI...")
        subprocess.run("streamlit run app/streamlit_app.py", shell=True)
    
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
