#!/usr/bin/env python3
"""
Quick check to see if the service is using the new API key
"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_service():
    print("🔍 Checking Gemini service...")
    
    # Check current API key
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"📋 Current API key: {api_key[:10] if api_key else 'None'}...")
    
    try:
        from gemini_service import gemini_service
        
        print(f"📊 Service available: {gemini_service.is_available()}")
        print(f"📋 Service API key: {gemini_service.api_key[:10] if gemini_service.api_key else 'None'}...")
        
        # Test with minimal data
        test_data = {
            'most_selling': [{'product_name': 'Test', 'units_sold': 10}],
            'insights': {}
        }
        
        print("🔄 Testing insights generation...")
        insights = gemini_service.generate_business_insights(test_data)
        
        if 'ai_insights' in insights:
            exec_summary = insights['ai_insights'].get('executive_summary', {})
            if 'critical_success_factors' in exec_summary and 'Gemini API setup required' in str(exec_summary):
                print("❌ Still using fallback insights")
                return False
            else:
                print("✅ Using real AI insights")
                return True
        else:
            print("❌ No insights generated")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    check_service()
