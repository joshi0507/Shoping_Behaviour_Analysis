# ProAnz Analytics - MongoDB Version

A comprehensive sales and product analytics dashboard powered by MongoDB, featuring real-time data streaming, advanced business insights, and predictive analytics.

## Features

### 🚀 Core Functionality
- **MongoDB Integration**: Scalable NoSQL database for efficient data storage and retrieval
- **Real-time Streaming**: Live sales data simulation with Socket.IO
- **Advanced Analytics**: Prophet-based sales forecasting and trend analysis
- **Business Intelligence**: AI-powered insights and recommendations
- **Interactive Visualizations**: Beautiful charts with Plotly.js

### 📊 Analytics & Insights
- **Product Performance Analysis**: Identify top and bottom performers
- **Pricing Strategy**: Optimize pricing based on elasticity analysis
- **Growth Opportunities**: Discover high-potential products and market trends
- **Inventory Management**: Smart recommendations for stock optimization
- **Marketing Insights**: Data-driven marketing strategies
- **Risk Analysis**: Identify and mitigate business risks

### 🎯 Business Intelligence
- **Executive Summary**: High-level overview with key metrics
- **Action Items**: Prioritized recommendations with timelines
- **Performance Categories**: Automatic product classification
- **Revenue Analysis**: Comprehensive revenue breakdown
- **Market Segmentation**: Customer and product segmentation insights

## Installation

### Prerequisites
- Python 3.8+
- MongoDB Atlas or local MongoDB instance
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ProAnz-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_mongo.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI="mongodb+srv://username:password@cluster.mongodb.net/"
   ```

4. **Run the application**
   ```bash
   python app_mongo.py
   ```

The application will be available at `http://127.0.0.1:5000`

## Usage

### Data Upload
1. **CSV Upload**: Upload a CSV file with columns: `product_name`, `date`, `units_sold`, `price`
2. **Manual Entry**: Enter data manually through the web interface
3. **Real-time Simulation**: Start live data streaming for testing

### Dashboard Features
- **Charts Tab**: Interactive visualizations of sales data
- **Insights Tab**: Comprehensive business analysis and recommendations
- **Actions Tab**: Prioritized action items with timelines

### API Endpoints
- `GET /`: Main dashboard
- `POST /upload`: Upload and analyze sales data
- Socket.IO events for real-time streaming

## File Structure

```
ProAnz-app/
├── app_mongo.py              # Main Flask application with MongoDB
├── analytics_mongo.py        # Business intelligence and insights module
├── requirements_mongo.txt    # Python dependencies
├── templates/
│   └── index_mongo.html     # Enhanced web interface
├── .env                     # Environment variables
└── README_MONGO.md          # This file
```

## MongoDB Schema

### Collections

#### sales_data
```javascript
{
  "_id": ObjectId,
  "product_name": "string",
  "date": ISODate,
  "units_sold": "number",
  "price": "number",
  "created_at": ISODate
}
```

#### daily_sales
```javascript
{
  "_id": ObjectId,
  "date": ISODate,
  "units_sold": "number",
  "updated_at": ISODate
}
```

## Analytics Features

### Business Insights Generated
1. **Executive Summary**: Overview of key performance indicators
2. **Product Performance**: Detailed analysis of product performance
3. **Pricing Strategy**: Recommendations for price optimization
4. **Growth Opportunities**: Identification of expansion opportunities
5. **Inventory Management**: Stock level recommendations
6. **Marketing Insights**: Data-driven marketing strategies
7. **Risk Analysis**: Business risk assessment and mitigation
8. **Action Items**: Prioritized recommendations with timelines

### Key Metrics Tracked
- Total products and revenue
- Average price and sales volume
- Top and bottom performing products
- Revenue leaders and underperformers
- Price elasticity and market trends
- Growth potential and market opportunities

## Development

### Adding New Analytics
1. Update `analytics_mongo.py` with new insight functions
2. Modify the `generate_insights()` function to include new analytics
3. Update the frontend in `index_mongo.html` to display new insights

### Database Operations
- All database operations use MongoDB aggregation pipelines for efficiency
- Data is automatically cleaned and validated before storage
- Historical data is maintained for trend analysis and forecasting

### Real-time Features
- Socket.IO enables live data streaming
- Background thread simulates real-time sales data
- Charts update automatically with new data

## Production Deployment

### Environment Setup
1. Set production environment variables
2. Configure MongoDB Atlas with proper security
3. Use Gunicorn for WSGI serving

### Docker Support
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_mongo.txt .
RUN pip install -r requirements_mongo.txt
COPY . .
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app_mongo:app"]
```

## Security Considerations
- Environment variables for sensitive data
- Input validation and sanitization
- MongoDB connection security
- CORS configuration for websockets

## Performance Optimization
- MongoDB aggregation pipelines for efficient queries
- Data limiting for large datasets
- Caching for frequently accessed data
- Background processing for heavy computations

## Troubleshooting

### Common Issues
1. **MongoDB Connection**: Check MONGO_URI in .env file
2. **Missing Dependencies**: Run `pip install -r requirements_mongo.txt`
3. **Port Conflicts**: Ensure port 5000 is available
4. **Data Format**: Verify CSV column names match requirements

### Logging
- Application logs: `proanz_mongo_app.log`
- MongoDB connection errors are logged
- Detailed error messages for debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the application logs
- Verify MongoDB connection and data format
