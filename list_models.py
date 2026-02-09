#!/usr/bin/env python3
"""
Script to list available Gemini models
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_available_models():
    """List all available Gemini models"""
    
    print("🔍 Listing available Gemini models...")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    try:
        import google.generativeai as genai
        
        # Configure API
        genai.configure(api_key=api_key)
        print("✅ Gemini configured")
        
        # List models
        print("📋 Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name} - {model.display_name}")
            else:
                print(f"❌ {model.name} - {model.display_name} (no generateContent)")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    list_available_models()
