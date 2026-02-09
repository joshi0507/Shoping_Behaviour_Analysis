#!/usr/bin/env python3
"""
Test script to verify Gemini API functionality
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_api():
    """Test Gemini API connection and functionality"""
    
    print("🧪 Testing Gemini API...")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        import google.generativeai as genai
        
        # Configure API
        genai.configure(api_key=api_key)
        print("✅ Gemini configured successfully")
        
        # Create model
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini model created")
        
        # Test simple generation
        print("🔄 Testing content generation...")
        response = model.generate_content("Hello! Please respond with 'API working' to confirm the connection.")
        
        print(f"✅ Gemini response: {response.text}")
        
        # Test with business data
        print("🔄 Testing business insights generation...")
        test_data = {
            "most_selling": [{"product_name": "Product A", "units_sold": 100}],
            "low_selling": [{"product_name": "Product B", "units_sold": 10}],
            "total_revenue": 5000
        }
        
        test_prompt = f"""
        Analyze this simple sales data and provide a brief business insight:
        {test_data}
        
        Respond with a single sentence business recommendation.
        """
        
        business_response = model.generate_content(test_prompt)
        print(f"✅ Business insight: {business_response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        return False

def test_service_integration():
    """Test the Gemini service integration"""
    
    print("\n🔧 Testing Gemini service integration...")
    
    try:
        from gemini_service import gemini_service
        
        print(f"✅ Service imported successfully")
        print(f"📊 Service available: {gemini_service.is_available()}")
        
        # Test with sample data
        test_analytics = {
            'most_selling': [{'product_name': 'Test Product', 'units_sold': 50}],
            'low_selling': [{'product_name': 'Low Product', 'units_sold': 5}],
            'daily_sales': [{'date': '2024-01-01', 'units_sold': 25}],
            'product_performance': [{'product_name': 'Test', 'units_sold': 50, 'price': 10.0}],
            'insights': {'test': 'data'}
        }
        
        print("🔄 Testing insights generation...")
        insights = gemini_service.generate_business_insights(test_analytics)
        
        if insights and 'ai_insights' in insights:
            print("✅ AI insights generated successfully")
            print(f"📝 Executive summary: {insights['ai_insights'].get('executive_summary', {}).get('title', 'N/A')}")
            return True
        else:
            print("❌ No AI insights generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing service: {str(e)}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Gemini API Test Suite")
    print("=" * 50)
    
    # Test 1: Direct API test
    api_test = test_gemini_api()
    
    # Test 2: Service integration test
    service_test = test_service_integration()
    
    print("\n📊 Test Results:")
    print(f"Direct API Test: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"Service Integration: {'✅ PASS' if service_test else '❌ FAIL'}")
    
    if api_test and service_test:
        print("\n🎉 All tests passed! Gemini AI is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting steps:")
        print("1. Verify your Gemini API key is valid")
        print("2. Check internet connection")
        print("3. Ensure google-generativeai package is installed")
        print("4. Check if API key has proper permissions")

if __name__ == "__main__":
    main()
