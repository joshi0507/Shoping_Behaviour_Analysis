# 🚀 ProAnz Analytics - Production Deployment Guide

## 📋 Overview
This guide covers deploying the ProAnz Analytics application to Render.com with zero impact on local development.

---

## 🏗️ Architecture

### **Environment-Based Configuration**
- **Development**: Uses local MongoDB, debug mode, `.env` file
- **Production**: Uses Render MongoDB Atlas, production settings, environment variables

### **Key Files Created**
- `config.py` - Environment-based configuration
- `render.yaml` - Render deployment configuration
- `.env.example` - Template for environment variables
- Updated `.gitignore` - Excludes sensitive files

---

## 🔧 Production Setup

### **1. Environment Variables (Render Dashboard)**
```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your_secure_random_key_here
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/proanz_analytics
GEMINI_API_KEY=your_production_gemini_api_key
```

### **2. MongoDB Atlas Setup**
1. **Go to MongoDB Atlas**
2. **Create new cluster**
3. **Get connection string**
4. **Add database user**
5. **Use connection string** in Render environment

---

## 📦 Deployment Steps

### **Step 1: Prepare Repository**
```bash
# Ensure all changes are committed
git add .
git commit -m "Add production deployment configuration"
git push origin main
```

### **Step 2: Render Deployment**
1. **Go to render.com**
2. **"New +" → "Web Service"**
3. **Connect GitHub repository**
4. **Use `render.yaml` configuration**
5. **Deploy automatically**

---

## 🔒 Security Configuration

### **Environment Variables Security**
- ✅ **`.env` excluded** from Git
- ✅ **Template provided** (`.env.example`)
- ✅ **Production secrets** separate from code
- ✅ **API keys** in environment only

### **CORS Configuration**
```python
# Production CORS (app.py)
if app.config.get('ENVIRONMENT') == 'production':
    CORS(app, origins=[
        "https://your-app-name.onrender.com",
        "https://www.your-app-name.onrender.com"
    ])
```

---

## 🔄 Local vs Production

### **Local Development (Unchanged)**
```bash
# Still works exactly as before
FLASK_ENV=development
FLASK_DEBUG=True
MONGO_URI=mongodb://localhost:27017/proanz_analytics
```

### **Production Deployment**
```bash
# Automatic on Render
FLASK_ENV=production
FLASK_DEBUG=False
MONGO_URI=mongodb+srv://prod-cluster.mongodb.net/proanz_analytics
```

---

## 🛠️ Configuration Files

### **config.py**
```python
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/proanz_analytics')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
class ProductionConfig(Config):
    DEBUG = False
    
class DevelopmentConfig(Config):
    DEBUG = True
```

### **render.yaml**
```yaml
services:
  type: web
  name: proanz-analytics
  env: python
  plan: free
  buildCommand: pip install -r requirements.txt
  startCommand: python app.py
  envVars:
    - key: FLASK_ENV
      value: production
    - key: FLASK_DEBUG
      value: False
  disk:
    name: proanz-uploads
    mountPath: /app/uploads
    sizeGB: 1
```

---

## 🌐 Production URLs

### **Application Access**
- **Main App**: `https://your-app-name.onrender.com`
- **Dashboard**: `https://your-app-name.onrender.com/dashboard?upload_id=xyz`
- **API**: `https://your-app-name.onrender.com/api/...`

### **Database Access**
- **MongoDB Atlas**: Via cluster dashboard
- **Connection**: Secure SRV connection string

---

## 📊 Features Status

### **✅ Production Ready**
- **Authentication** → Fully functional
- **Upload System** → CSV and manual data
- **Analytics Engine** → MongoDB-based
- **AI Insights** → Gemini integration
- **Dashboard Pages** → Separate analytics views
- **Upload History** → Complete with navigation
- **CORS** → Configured for production
- **Security** → Environment-based secrets

### **🔄 Local Development**
- **Zero Impact** → All local features unchanged
- **Same Codebase** → Single codebase for both environments
- **Easy Switching** → Environment variable based

---

## 🚨 Troubleshooting

### **Common Issues**
1. **MongoDB Connection**
   - Check connection string in Render environment
   - Verify MongoDB Atlas network access

2. **CORS Issues**
   - Update origins in `app.py`
   - Verify frontend URLs

3. **API Key Issues**
   - Ensure Gemini API key is valid
   - Check environment variables in Render

4. **Upload Issues**
   - Check disk space in Render
   - Verify file permissions

### **Debug Mode**
```python
# Production debugging (temporary)
# Add to render.yaml envVars:
- key: FLASK_DEBUG
  value: true
```

---

## 📈 Scaling

### **Render Free Tier**
- **RAM**: 512MB - 1GB
- **CPU**: Shared
- **Storage**: 1GB (configurable)
- **Bandwidth**: 100GB/month

### **Upgrade Path**
1. **Starter** → Better performance
2. **Standard** → More RAM/CPU
3. **Pro** → Dedicated resources

---

## 🔍 Monitoring

### **Render Dashboard**
- **Logs**: Real-time application logs
- **Metrics**: Performance monitoring
- **Deployments**: Rollback history
- **Environment**: Variable management

### **Application Logs**
```bash
# View in Render dashboard or via SSH
tail -f proanz_mongo_app.log
```

---

## 🎯 Success Checklist

- [x] **Config files created** (`config.py`, `render.yaml`)
- [x] **Environment variables configured**
- [x] **CORS setup for production**
- [x] **Security files excluded** (`.gitignore`)
- [x] **Dependencies updated** (`requirements.txt`)
- [x] **MongoDB Atlas ready**
- [x] **Local development preserved**
- [x] **Deployment documentation complete**

---

## 🚀 Ready for Production!

Your ProAnz Analytics application is now fully configured for production deployment on Render with zero impact on local development.
