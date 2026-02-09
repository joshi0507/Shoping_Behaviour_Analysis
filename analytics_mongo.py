# -*- coding: utf-8 -*-
# analytics_mongo.py - Advanced analytics and business insights module
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def generate_insights(prod_sales_df, raw_sales_df):
    """
    Generate comprehensive business insights and recommendations
    
    Args:
        prod_sales_df: Aggregated product sales data
        raw_sales_df: Raw sales data with dates
    
    Returns:
        dict: Comprehensive insights and recommendations (JSON-serializable)
    """
    try:
        insights = {}
        
        # Convert DataFrames to dictionaries for JSON serialization
        prod_sales_dict = prod_sales_df.to_dict('records') if not prod_sales_df.empty else []
        raw_sales_dict = raw_sales_df.to_dict('records') if not raw_sales_df.empty else []
        
        # Basic Statistics
        total_products = len(prod_sales_df)
        total_revenue = (prod_sales_df['units_sold'] * prod_sales_df['price']).sum()
        total_units = prod_sales_df['units_sold'].sum()
        avg_price = prod_sales_df['price'].mean()
        avg_units_per_product = prod_sales_df['units_sold'].mean()
        
        # Top and Bottom Performers
        if not prod_sales_df.empty:
            top_performer = prod_sales_df.loc[prod_sales_df['units_sold'].idxmax()]
            bottom_performer = prod_sales_df.loc[prod_sales_df['units_sold'].idxmin()]
            
            # Convert to regular Python types
            top_performer_dict = {
                'product_name': str(top_performer['product_name']),
                'units_sold': float(top_performer['units_sold']),
                'price': float(top_performer['price'])
            }
            bottom_performer_dict = {
                'product_name': str(bottom_performer['product_name']),
                'units_sold': float(bottom_performer['units_sold']),
                'price': float(bottom_performer['price'])
            }
        else:
            top_performer_dict = {'product_name': 'N/A', 'units_sold': 0, 'price': 0}
            bottom_performer_dict = {'product_name': 'N/A', 'units_sold': 0, 'price': 0}
        
        # Revenue Analysis
        prod_sales_df['revenue'] = prod_sales_df['units_sold'] * prod_sales_df['price']
        if not prod_sales_df.empty:
            top_revenue_product = prod_sales_df.loc[prod_sales_df['revenue'].idxmax()]
            top_revenue_dict = {
                'product_name': str(top_revenue_product['product_name']),
                'revenue': float(top_revenue_product['revenue'])
            }
        else:
            top_revenue_dict = {'product_name': 'N/A', 'revenue': 0}
        
        # Price Segmentation
        if not prod_sales_df.empty:
            price_quartiles = prod_sales_df['price'].quantile([0.25, 0.5, 0.75])
            low_price_threshold = float(price_quartiles[0.25])
            high_price_threshold = float(price_quartiles[0.75])
            
            low_price_products = prod_sales_df[prod_sales_df['price'] <= low_price_threshold]
            high_price_products = prod_sales_df[prod_sales_df['price'] >= high_price_threshold]
            
            low_price_products_list = low_price_products.to_dict('records')
            high_price_products_list = high_price_products.to_dict('records')
        else:
            low_price_threshold = 0
            high_price_threshold = 0
            low_price_products_list = []
            high_price_products_list = []
        
        # Performance Categories
        if not prod_sales_df.empty:
            prod_sales_df['performance_category'] = pd.cut(
                prod_sales_df['units_sold'],
                bins=[0, prod_sales_df['units_sold'].quantile(0.33), 
                      prod_sales_df['units_sold'].quantile(0.67), float('inf')],
                labels=['Low Performer', 'Medium Performer', 'High Performer']
            )
            
            high_performers = prod_sales_df[prod_sales_df['performance_category'] == 'High Performer']
            low_performers = prod_sales_df[prod_sales_df['performance_category'] == 'Low Performer']
            
            high_performers_list = high_performers.to_dict('records')
            low_performers_list = low_performers.to_dict('records')
        else:
            high_performers_list = []
            low_performers_list = []
        
        # Generate Insights
        insights['executive_summary'] = generate_executive_summary(
            total_products, total_revenue, total_units, top_performer_dict, bottom_performer_dict
        )
        
        insights['product_performance'] = generate_product_insights(
            prod_sales_dict, top_performer_dict, bottom_performer_dict, top_revenue_dict
        )
        
        insights['pricing_strategy'] = generate_pricing_insights(
            prod_sales_df, low_price_products, high_price_products, avg_price
        )
        
        insights['growth_opportunities'] = generate_growth_insights(
            prod_sales_df, raw_sales_df
        )
        
        insights['inventory_recommendations'] = generate_inventory_insights(
            prod_sales_df
        )
        
        insights['marketing_insights'] = generate_marketing_insights(
            prod_sales_df, low_price_products, high_price_products
        )
        
        insights['risk_analysis'] = generate_risk_insights(
            prod_sales_df
        )
        
        insights['action_items'] = generate_action_items(
            prod_sales_df, top_performer_dict, bottom_performer_dict
        )
        
        logger.info('Generated comprehensive business insights')
        return insights
        
    except Exception as e:
        logger.error(f'Error generating insights: {str(e)}')
        return {'error': 'Failed to generate insights'}

def generate_executive_summary(total_products, total_revenue, total_units, top_performer, bottom_performer):
    """Generate executive summary"""
    return {
        'title': 'Executive Summary',
        'content': f"""
        Your portfolio contains {total_products} products with total revenue of ${total_revenue:,.2f} 
        from {total_units:,.0f} units sold. {top_performer['product_name']} is your star performer 
        with {top_performer['units_sold']:,.0f} units sold, while {bottom_performer['product_name']} 
        needs immediate attention with only {bottom_performer['units_sold']:,.0f} units sold.
        """,
        'key_metrics': [
            f'Total Products: {total_products}',
            f'Total Revenue: ${total_revenue:,.2f}',
            f'Total Units Sold: {total_units:,.0f}',
            f'Top Performer: {top_performer["product_name"]}',
            f'Average Price: ${top_performer["price"]:.2f}'
        ]
    }

def generate_product_insights(prod_sales_dict, top_performer, bottom_performer, top_revenue_product):
    """Generate product-specific insights"""
    
    # Count performers from dict
    high_performers_count = len([p for p in prod_sales_dict if p.get('performance_category') == 'High Performer'])
    low_performers_count = len([p for p in prod_sales_dict if p.get('performance_category') == 'Low Performer'])
    
    return {
        'title': 'Product Performance Analysis',
        'top_performer': {
            'product': top_performer['product_name'],
            'units_sold': top_performer['units_sold'],
            'revenue': top_performer['units_sold'] * top_performer['price'],
            'insight': f"{top_performer['product_name']} dominates with {top_performer['units_sold']:,.0f} units sold. "
                      f"Consider expanding this product line or creating similar variants."
        },
        'bottom_performer': {
            'product': bottom_performer['product_name'],
            'units_sold': bottom_performer['units_sold'],
            'revenue': bottom_performer['units_sold'] * bottom_performer['price'],
            'insight': f"{bottom_performer['product_name']} is underperforming with only {bottom_performer['units_sold']:,.0f} units. "
                      f"Review pricing, marketing, or consider discontinuation."
        },
        'revenue_leader': {
            'product': top_revenue_product['product_name'],
            'revenue': top_revenue_product['revenue'],
            'insight': f"{top_revenue_product['product_name']} generates the most revenue at ${top_revenue_product['revenue']:,.2f}. "
                      f"Protect this cash cow while exploring growth opportunities."
        },
        'performance_distribution': {
            'high_performers': high_performers_count,
            'medium_performers': len(prod_sales_dict) - high_performers_count - low_performers_count,
            'low_performers': low_performers_count,
            'insight': f"You have {high_performers_count} high performers, {low_performers_count} low performers. "
                      f"Focus on converting medium performers to high performers."
        }
    }

def generate_pricing_insights(prod_sales_dict, low_price_products, high_price_products, avg_price):
    """Generate pricing strategy insights"""
    
    # Calculate averages from dictionaries
    low_price_avg_sales = sum(p['units_sold'] for p in low_price_products) / len(low_price_products) if low_price_products else 0
    high_price_avg_sales = sum(p['units_sold'] for p in high_price_products) / len(high_price_products) if high_price_products else 0
    
    # Find opportunity products
    high_potential_products = [p for p in prod_sales_dict if p.get('units_sold', 0) > avg_price and p.get('price', 0) < avg_price]
    low_price_high_sales = [p for p in prod_sales_dict if p.get('units_sold', 0) > avg_price and p.get('price', 0) <= avg_price]
    
    return {
        'title': 'Pricing Strategy Analysis',
        'average_price': avg_price,
        'price_elasticity': {
            'insight': f"Low-price products average {low_price_avg_sales:.0f} units vs high-price at {high_price_avg_sales:.0f} units. "
                      f"Consider strategic price adjustments for optimal revenue.",
            'recommendation': 'Test price increases on popular low-price items and promotions on high-price items.'
        },
        'opportunity_products': {
            'high_price_low_sales': [p for p in prod_sales_dict if p.get('price', 0) > avg_price and p.get('units_sold', 0) < avg_price],
            'low_price_high_sales': low_price_high_sales,
            'insight': 'Identify products with pricing misalignment that could benefit from optimization.'
        }
    }

def generate_growth_insights(prod_sales_dict, raw_sales_dict):
    """Generate growth opportunity insights"""
    
    # Find top 3 high potential products
    avg_units = sum(p['units_sold'] for p in prod_sales_dict) / len(prod_sales_dict) if prod_sales_dict else 0
    avg_price = sum(p['price'] for p in prod_sales_dict) / len(prod_sales_dict) if prod_sales_dict else 0
    
    high_potential = [p for p in prod_sales_dict 
                    if p.get('units_sold', 0) > avg_units and p.get('price', 0) < avg_price][:3]
    
    return {
        'title': 'Growth Opportunities',
        'market_trend': 'Sales data analysis completed',
        'high_potential_products': {
            'count': len(high_potential),
            'products': [p['product_name'] for p in high_potential],
            'insight': 'These products combine good sales volume with competitive pricing - excellent growth candidates.'
        },
        'expansion_opportunities': {
            'insight': 'Consider product line extensions, bundle deals, or market expansion for top performers.',
            'strategies': [
                'Create premium versions of popular products',
                'Develop complementary product bundles',
                'Explore new market segments',
                'Implement referral programs'
            ]
        }
    }

def generate_inventory_insights(prod_sales_dict):
    """Generate inventory management insights"""
    
    # Calculate sales velocity
    max_units = max(p['units_sold'] for p in prod_sales_dict) if prod_sales_dict else 1
    
    fast_moving = [p for p in prod_sales_dict if p.get('units_sold', 0) > 0.7 * max_units]
    slow_moving = [p for p in prod_sales_dict if p.get('units_sold', 0) < 0.3 * max_units]
    
    return {
        'title': 'Inventory Management Recommendations',
        'fast_moving_products': {
            'count': len(fast_moving),
            'products': [p['product_name'] for p in fast_moving],
            'recommendation': 'Increase safety stock and ensure supply chain reliability for these products.'
        },
        'slow_moving_products': {
            'count': len(slow_moving),
            'products': [p['product_name'] for p in slow_moving],
            'recommendation': 'Reduce inventory levels, consider clearance promotions, or bundle with popular items.'
        },
        'inventory_optimization': {
            'insight': 'Implement just-in-time inventory for slow movers and bulk purchasing for fast movers.',
            'action_items': [
                'Set up automated reorder points',
                'Implement ABC analysis (Always Better Control)',
                'Consider dropshipping for low-volume items',
                'Optimize warehouse layout based on sales velocity'
            ]
        }
    }

def generate_marketing_insights(prod_sales_dict, low_price_products, high_price_products):
    """Generate marketing strategy insights"""
    return {
        'title': 'Marketing Strategy Recommendations',
        'segmentation_strategy': {
            'insight': 'Tailor marketing messages by product performance and price segments.',
            'recommendations': [
                'Highlight value proposition for budget-conscious customers',
                'Emphasize quality and premium features for high-end products',
                'Create tiered marketing campaigns for different price segments',
                'Use social proof for popular products'
            ]
        },
        'promotional_opportunities': {
            'bundle_deals': 'Create bundles combining popular and less popular products',
            'seasonal_campaigns': 'Align promotions with seasonal demand patterns',
            'loyalty_programs': 'Reward repeat customers of high-performing products',
            'cross_selling': 'Promote complementary products during checkout'
        },
        'channel_optimization': {
            'insight': 'Focus marketing budget on channels that deliver highest ROI for each product segment.',
            'recommendations': [
                'Invest in SEO for high-performing products',
                'Use social media ads for visually appealing products',
                'Implement email marketing for repeat purchases',
                'Consider influencer partnerships for premium products'
            ]
        }
    }

def generate_risk_insights(prod_sales_dict):
    """Generate risk analysis insights"""
    
    if not prod_sales_dict:
        return {'title': 'Risk Analysis', 'error': 'No data available for analysis'}
    
    # Calculate concentration risk
    total_units = sum(p['units_sold'] for p in prod_sales_dict)
    top_3_units = sum(sorted([p['units_sold'] for p in prod_sales_dict], reverse=True)[:3])
    top_3_concentration = top_3_units / total_units if total_units > 0 else 0
    
    # Identify at-risk products
    avg_price = sum(p['price'] for p in prod_sales_dict) / len(prod_sales_dict)
    at_risk = [p for p in prod_sales_dict 
                if p.get('units_sold', 0) < total_units / len(prod_sales_dict) * 0.25 
                and p.get('price', 0) > avg_price]
    
    return {
        'title': 'Risk Analysis and Mitigation',
        'concentration_risk': {
            'level': 'High' if top_3_concentration > 0.6 else 'Medium' if top_3_concentration > 0.4 else 'Low',
            'percentage': f"{top_3_concentration:.1%}",
            'insight': f"Top 3 products represent {top_3_concentration:.1%} of total sales. "
                      f"Diversify product portfolio to reduce dependency risk."
        },
        'at_risk_products': {
            'count': len(at_risk),
            'products': [p['product_name'] for p in at_risk],
            'insight': 'These products combine low sales with high prices - consider price adjustments or enhanced marketing.'
        },
        'mitigation_strategies': [
            'Develop contingency plans for supply chain disruptions',
            'Maintain buffer inventory for critical products',
            'Diversify supplier base',
            'Monitor market trends and competitor activities',
            'Implement product lifecycle management'
        ]
    }

def generate_action_items(prod_sales_dict, top_performer, bottom_performer):
    """Generate prioritized action items"""
    return {
        'title': 'Immediate Action Items',
        'priority_1': [
            f"Analyze and optimize {bottom_performer['product_name']} performance",
            "Review pricing strategy across product portfolio",
            "Implement inventory management system",
            "Set up sales performance monitoring dashboard"
        ],
        'priority_2': [
            f"Expand {top_performer['product_name']} product line",
            "Develop marketing campaigns for medium performers",
            "Create product bundles and promotional offers",
            "Establish supplier relationships for fast-moving items"
        ],
        'priority_3': [
            "Explore new market segments",
            "Implement customer loyalty programs",
            "Develop e-commerce optimization strategies",
            "Create data-driven forecasting models"
        ],
        'timeline': {
            'immediate': 'Focus on bottom performer analysis and pricing review',
            '30_days': 'Launch marketing campaigns and optimize inventory',
            '90_days': 'Expand product lines and explore new markets'
        }
    }
