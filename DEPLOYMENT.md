# 🚀 Deployment Guide for Render

## Quick Deployment Steps:

### 1. Create Render Account
1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with your GitHub account

### 2. Deploy Your App
1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `Crop-profit-prediction`
3. Configure the service:
   - **Name**: `crop-profit-predictor` (or any name you like)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Runtime**: `Python 3`
   - **Build Command**: `./build.sh` or `pip install -r requirements.txt && python train.py`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free

4. Click "Create Web Service"

### 3. Wait for Deployment (5-10 minutes)
- Render will:
  - Install all dependencies
  - Train your ML models
  - Start the Flask app
  
### 4. Access Your App
- Once deployed, Render will give you a URL like:
  - `https://crop-profit-predictor-xxxx.onrender.com`
- Click it to see your live app! 🎉

## Important Notes:

⚠️ **Free Tier Limitations:**
- App sleeps after 15 minutes of inactivity
- Takes ~30 seconds to wake up on first request
- 750 hours/month free runtime

💡 **Tips:**
- First deployment takes longer (builds ML models)
- Subsequent deployments are faster
- Auto-deploys on every git push to main branch

## Troubleshooting:

If build fails:
1. Check the build logs in Render dashboard
2. Common issues:
   - Python version mismatch → Set to Python 3.11
   - Missing dependencies → Check requirements.txt
   - Training timeout → Increase build timeout in settings

## Upgrade to Paid Plan (Optional):

For production use:
- **Starter Plan**: $7/month
  - No sleep time
  - Faster performance
  - Custom domain support

---

## Alternative: Quick Deploy Button

You can also add this to your README.md:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Samarth-2003-web/Crop-profit-prediction)

---

## After Deployment:

✅ Your app will be live at: `https://your-app-name.onrender.com`
✅ Share the link with farmers and users!
✅ Monitor usage in Render dashboard
✅ Check logs for any errors

Good luck with your deployment! 🌾🚀
