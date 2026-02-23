# 🚀 Urban Pulse 2.0 - Backend Deployment Guide

## 📋 Step-by-Step Render Deployment

### 1. Sign Up for Render
1. Go to https://render.com
2. Click "Sign Up" → "Sign up with GitHub"
3. Authorize Render to access your GitHub repositories
4. Select your free plan

### 2. Create New Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `monishaasenthil/UrbanPulse2.0`
3. Configure the service:

**Basic Settings:**
- **Name**: `urban-pulse-api`
- **Root Directory**: `/`
- **Runtime**: `Python 3`
- **Instance Type**: `Free`

**Build Settings:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m api.app`

**Environment Variables:**
- `FLASK_ENV`: `production`
- `PORT`: `5000`

### 3. Deploy!
Click "Create Web Service" and Render will:
- Clone your repository
- Install dependencies
- Start the Flask application
- Provide you with a URL like: `https://urban-pulse-api.onrender.com`

### 4. Update Frontend API URL
Once your backend is deployed, update the frontend:

1. Go to your Vercel dashboard
2. Find your project: `urbanpulse-2-0`
3. Go to "Settings" → "Environment Variables"
4. Add/Update:
   - **Variable**: `REACT_APP_API_URL`
   - **Value**: `https://urban-pulse-api.onrender.com/api`
5. Redeploy the frontend

### 5. Test the Integration
- Backend: `https://urban-pulse-api.onrender.com/api/health`
- Frontend: `https://urbanpulse-2-0.vercel.app`

## 🎯 Expected URLs After Deployment
- **Backend API**: `https://urban-pulse-api.onrender.com`
- **Frontend**: `https://urbanpulse-2-0.vercel.app`

## 🔧 Troubleshooting

### If Build Fails:
1. Check the build logs in Render dashboard
2. Make sure `requirements.txt` has all dependencies
3. Verify `api.app` can be imported as a module

### If API Doesn't Respond:
1. Check the service logs in Render
2. Make sure the Flask app is listening on the correct port
3. Verify CORS is configured properly

### If Frontend Can't Connect:
1. Check the API URL in Vercel environment variables
2. Make sure the backend is deployed and running
3. Test the API endpoint directly in browser

## 📊 What You'll Get
- ✅ 24/7 running backend API
- ✅ Real-time traffic risk data
- ✅ ML model predictions
- ✅ Weather integration
- ✅ Complete deployed system

## 🎉 Success!
Once deployed, your Urban Pulse 2.0 system will be fully functional with:
- Live risk scoring
- Real-time incident monitoring
- Intelligent signal control
- Emergency routing
- Comprehensive dashboard

The system will be ready for city traffic departments to use!
