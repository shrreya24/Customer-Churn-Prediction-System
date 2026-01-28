# 🚀 SUPER SIMPLE GUIDE - How to Run This Project

Follow these steps **EXACTLY** in order:

## Step 1: Install Dependencies
```bash
cd "/home/apex/Projects - ML/Customer retention Project"
pip3 install -r requirements.txt
```
**Wait for this to finish!** It will take 2-3 minutes.

---

## Step 2: Set Up Kaggle (for data download)

### Option A - If you have Kaggle account:
1. Go to https://www.kaggle.com and log in
2. Click your profile picture → **Account**
3. Scroll to **API** → Click **Create New API Token**
4. A file `kaggle.json` will download
5. Run these commands:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option B - If you DON'T have Kaggle account:
I can help you download the dataset manually. Let me know!

---

## Step 3: Run the System

### Easiest Way:
```bash
python3 quickstart.py
```
Then choose **option 4** (Quick Demo)

### OR Step-by-Step:
```bash
# 1. Download data (needs Kaggle API)
python3 src/data/ingestion.py

# 2. Process data
python3 src/data/preprocessing.py

# 3. Train the main model (XGBoost)
python3 src/models/tree_models.py

# 4. Launch the web app
streamlit run app/streamlit_app.py
```

---

## Step 4: Open the Web App

After running Streamlit, it will show:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open that URL in your browser!

---

## 🆘 If You Get Errors:

**"streamlit: not found"**
```bash
pip3 install streamlit
```

**"No module named 'pandas'"** (or any library)
```bash
pip3 install pandas numpy scikit-learn xgboost
```

**Kaggle API error**
- You need to set up Kaggle credentials (see Step 2)
- OR I can help you download data manually

---

## ✅ What You Should See:

1. After Step 1: "Successfully installed..." messages
2. After data download: "Dataset downloaded successfully"
3. After model training: "ROC-AUC: 0.85" or similar metrics
4. After Streamlit: A web page opens with the prediction interface

---

## 🎯 Fastest Test (No Kaggle Needed):

If you want to skip data download and just see the UI:
```bash
# Just launch the UI (won't work for predictions without models)
streamlit run app/streamlit_app.py
```

But you'll need to train models first for it to work properly!

---

**Need help?** Let me know which step fails and I'll help you fix it!
