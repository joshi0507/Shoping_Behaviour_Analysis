"""
Google Gemini AI Service for Business Insights Generation
This module handles integration with Google Gemini API to generate business insights
from sales analytics data and graphs.
"""

import os
import json
import logging
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

class GeminiInsightsService:
    """Service for generating business insights using Google Gemini AI"""
    
    def __init__(self):
        """Initialize Gemini service with API key"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"Initializing Gemini service - API key found: {bool(self.api_key)}")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            self.model = None
        else:
            try:
                import google.generativeai as genai
                logger.info("Configuring Gemini with API key...")
                genai.configure(api_key=self.api_key)
                
                # Test the API key by creating a model
                logger.info("Creating Gemini model...")
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Test if model is working by checking its attributes
                if hasattr(self.model, 'generate_content'):
                    logger.info("Gemini AI service initialized successfully")
                else:
                    logger.error("Gemini model created but missing generate_content method")
                    self.model = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize Gemini service: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.model is not None
    
    def generate_business_insights(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive business insights using Gemini AI
        
        Args:
            analytics_data: Dictionary containing graph data and analysis results
            
        Returns:
            Dictionary containing AI-generated business insights
        """
        if not self.is_available():
            logger.warning("Gemini service not available, returning default insights")
            return self._get_default_insights()
        
        try:
            # Prepare comprehensive prompt with analytics data
            prompt = self._build_insights_prompt(analytics_data)
            logger.info("Sending request to Gemini AI...")
            
            # Generate insights using Gemini
            response = self.model.generate_content(prompt)
            logger.info(f"Received Gemini response: {response.text[:200]}...")
            
            # Parse and structure the response
            insights = self._parse_gemini_response(response.text)
            
            logger.info("Successfully generated Gemini business insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating Gemini insights: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return self._get_default_insights()
    
    def _build_insights_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive prompt for Gemini AI"""
        
        prompt = """
        You are a senior business analyst and data scientist expert. Analyze the following sales analytics data and provide comprehensive business insights.
        
        SALES ANALYTICS DATA:
        {analytics_data}
        
        Please provide detailed business insights covering these areas:
        
        1. **PERFORMANCE ANALYSIS**
           - Top performing products and their success factors
           - Underperforming products and improvement opportunities
           - Revenue trends and patterns
           - Product portfolio optimization recommendations
        
        2. **MARKET INSIGHTS**
           - Customer behavior patterns
           - Market trends and opportunities
           - Competitive positioning insights
           - Growth potential areas
        
        3. **STRATEGIC RECOMMENDATIONS**
           - Short-term actions (next 30 days)
           - Medium-term strategies (next 90 days)
           - Long-term initiatives (next 6-12 months)
           - Risk mitigation strategies
        
        4. **FINANCIAL INSIGHTS**
           - Revenue optimization opportunities
           - Cost reduction strategies
           - Pricing strategy recommendations
           - ROI improvement suggestions
        
        5. **OPERATIONAL EFFICIENCY**
           - Inventory management insights
           - Supply chain optimization
           - Process improvement recommendations
           - Technology integration opportunities
        
        Format your response as structured JSON with the following schema:
        {{
            "ai_insights": {{
                "performance_analysis": {{
                    "title": "Performance Analysis",
                    "summary": "Overall performance summary",
                    "top_performers": ["Product 1", "Product 2"],
                    "improvement_areas": ["Area 1", "Area 2"],
                    "key_insights": ["Insight 1", "Insight 2"]
                }},
                "market_insights": {{
                    "title": "Market Insights",
                    "summary": "Market analysis summary",
                    "trends": ["Trend 1", "Trend 2"],
                    "opportunities": ["Opportunity 1", "Opportunity 2"],
                    "recommendations": ["Recommendation 1", "Recommendation 2"]
                }},
                "strategic_recommendations": {{
                    "title": "Strategic Recommendations",
                    "immediate_actions": ["Action 1", "Action 2"],
                    "short_term_strategies": ["Strategy 1", "Strategy 2"],
                    "long_term_initiatives": ["Initiative 1", "Initiative 2"]
                }},
                "financial_insights": {{
                    "title": "Financial Insights",
                    "revenue_optimization": ["Optimization 1", "Optimization 2"],
                    "cost_reduction": ["Reduction 1", "Reduction 2"],
                    "pricing_strategy": ["Pricing 1", "Pricing 2"]
                }},
                "operational_efficiency": {{
                    "title": "Operational Efficiency",
                    "inventory_insights": ["Insight 1", "Insight 2"],
                    "process_improvements": ["Improvement 1", "Improvement 2"],
                    "technology_opportunities": ["Opportunity 1", "Opportunity 2"]
                }},
                "executive_summary": {{
                    "title": "Executive Summary",
                    "key_takeaways": ["Takeaway 1", "Takeaway 2"],
                    "critical_success_factors": ["Factor 1", "Factor 2"],
                    "next_steps": ["Step 1", "Step 2"]
                }}
            }}
        }}
        
        Make insights actionable, specific, and data-driven. Focus on practical business value.
        """.format(analytics_data=json.dumps(data, indent=2, default=str))
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and ensure proper JSON structure"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                insights = json.loads(json_str)
                return insights
            else:
                # Fallback: create structured insights from text
                return self._create_insights_from_text(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._create_insights_from_text(response_text)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return self._get_default_insights()
    
    def _create_insights_from_text(self, text: str) -> Dict[str, Any]:
        """Create structured insights from unstructured text response"""
        return {
            "ai_insights": {
                "performance_analysis": {
                    "title": "Performance Analysis",
                    "summary": text[:500] + "..." if len(text) > 500 else text,
                    "top_performers": ["Analysis based on data"],
                    "improvement_areas": ["Review detailed analysis"],
                    "key_insights": [text[:200] + "..." if len(text) > 200 else text]
                },
                "market_insights": {
                    "title": "Market Insights",
                    "summary": "AI-generated market analysis",
                    "trends": ["Data-driven trends identified"],
                    "opportunities": ["Growth opportunities detected"],
                    "recommendations": ["Strategic recommendations available"]
                },
                "strategic_recommendations": {
                    "title": "Strategic Recommendations",
                    "immediate_actions": ["Review AI analysis"],
                    "short_term_strategies": ["Implement data-driven strategies"],
                    "long_term_initiatives": ["Long-term planning recommended"]
                },
                "financial_insights": {
                    "title": "Financial Insights",
                    "revenue_optimization": ["Optimization opportunities identified"],
                    "cost_reduction": ["Cost-saving measures available"],
                    "pricing_strategy": ["Pricing strategy recommendations"]
                },
                "operational_efficiency": {
                    "title": "Operational Efficiency",
                    "inventory_insights": ["Inventory optimization possible"],
                    "process_improvements": ["Process enhancements identified"],
                    "technology_opportunities": ["Technology integration opportunities"]
                },
                "executive_summary": {
                    "title": "Executive Summary",
                    "key_takeaways": ["AI-generated insights available"],
                    "critical_success_factors": ["Data-driven success factors"],
                    "next_steps": ["Implement recommended actions"]
                }
            }
        }
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Get default insights when Gemini is not available"""
        return {
            "ai_insights": {
                "performance_analysis": {
                    "title": "Performance Analysis",
                    "summary": "AI insights not available. Please configure Gemini API key.",
                    "top_performers": ["Configure Gemini API for detailed analysis"],
                    "improvement_areas": ["Add GEMINI_API_KEY to environment variables"],
                    "key_insights": ["AI-powered insights require Gemini configuration"]
                },
                "market_insights": {
                    "title": "Market Insights",
                    "summary": "Advanced market insights available with Gemini AI",
                    "trends": ["Configure API for trend analysis"],
                    "opportunities": ["AI opportunities analysis available"],
                    "recommendations": ["Set up Gemini for recommendations"]
                },
                "strategic_recommendations": {
                    "title": "Strategic Recommendations",
                    "immediate_actions": ["Configure Gemini API key"],
                    "short_term_strategies": ["Enable AI-powered analytics"],
                    "long_term_initiatives": ["Implement AI-driven strategy"]
                },
                "financial_insights": {
                    "title": "Financial Insights",
                    "revenue_optimization": ["AI optimization available"],
                    "cost_reduction": ["AI cost analysis available"],
                    "pricing_strategy": ["AI pricing insights available"]
                },
                "operational_efficiency": {
                    "title": "Operational Efficiency",
                    "inventory_insights": ["AI inventory analysis available"],
                    "process_improvements": ["AI process optimization available"],
                    "technology_opportunities": ["AI tech integration available"]
                },
                "executive_summary": {
                    "title": "Executive Summary",
                    "key_takeaways": ["AI insights require configuration"],
                    "critical_success_factors": ["Gemini API setup required"],
                    "next_steps": ["Configure and restart application"]
                }
            }
        }

# Global instance
gemini_service = GeminiInsightsService()
