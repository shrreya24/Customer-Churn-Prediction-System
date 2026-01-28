# Kaggle API Setup Guide

## Your Kaggle API Token
You have: `KGAT_555d69f296fff7af73ea14cbf012a5ab`

## What's Needed

To use Kaggle API, you need TWO pieces of information:
1. **Your Kaggle Username** (e.g., "john_doe")
2. **Your API Key** (you already have this)

## Option 1: Setup with Your Username

If you tell me your Kaggle username, I'll create the `~/.kaggle/kaggle.json` file like this:

```json
{
  "username": "YOUR_USERNAME_HERE",
  "key": "KGAT_555d69f296fff7af73ea14cbf012a5ab"
}
```

## Option 2: Download Manually (Recommended - Network Issues)

Since you're experiencing network timeouts, manual download is faster:

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click the **Download** button (it's a small 200KB zip file)
3. Extract the ZIP file
4. Move `WA_Fn-UseC_-Telco-Customer-Churn.csv` to:
   ```
   /home/apex/Projects - ML/Customer retention Project/data/raw/
   ```

Then you can skip the data ingestion step and go straight to:
```bash
python3 src/data/preprocessing.py
```

## Next Steps

Tell me:
- Your Kaggle username, OR  
- Type "manual" if you want to download the file yourself
