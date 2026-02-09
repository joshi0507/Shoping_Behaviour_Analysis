#!/usr/bin/env python3
"""
Setup script for Google Gemini AI integration
This script helps verify and set up the Gemini API integration.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    packages = [
        "google-generativeai==0.8.3",
        "requests>=2.25.0",
        "protobuf>=3.20.0"
    ]
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing {package}: {str(e)}")

def check_environment():
    """Check environment setup"""
    print("\n🔍 Checking environment setup...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        create_env_file()
    
    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ GEMINI_API_KEY found in environment")
    else:
        print("❌ GEMINI_API_KEY not found in environment")
        print("📝 Please add your Gemini API key to .env file:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        print("\n🔗 Get your API key from: https://aistudio.google.com/app/apikey")

def create_env_file():
    """Create .env file with template"""
    env_content = """# ProAnz Analytics Environment Variables

# MongoDB Configuration
MONGO_URI="mongodb+srv://username:password@cluster.mongodb.net/"

# Flask Configuration
SECRET_KEY="your-secret-key-here-change-in-production"

# Google Gemini AI Configuration
GEMINI_API_KEY="your_gemini_api_key_here"

# Get your Gemini API key from: https://aistudio.google.com/app/apikey
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env file with template")

def test_gemini_import():
    """Test if Gemini can be imported"""
    print("\n🧪 Testing Gemini import...")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI package imported successfully")
        
        # Test basic functionality
        if hasattr(genai, 'GenerativeModel'):
            print("✅ GenerativeModel class available")
        else:
            print("⚠️  GenerativeModel class not found")
            
    except ImportError as e:
        print(f"❌ Failed to import Google Generative AI: {str(e)}")
        print("💡 Try running: pip install google-generativeai==0.8.3")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 ProAnz Analytics - Gemini AI Setup")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Check environment
    check_environment()
    
    # Test import
    if test_gemini_import():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Add your Gemini API key to .env file")
        print("2. Run the application: python app.py")
        print("3. Upload data to see AI-powered insights")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
