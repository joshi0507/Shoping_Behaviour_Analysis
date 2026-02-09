# ProAnz Analytics – Shopping Behaviour Analysis

**A modern, real-time sales analytics dashboard with forecasting and AI-powered business recommendations**


*Analysis Dashboard*

Register page

<img width="220" height="300" alt="image" src="https://github.com/user-attachments/assets/04725493-4514-4da0-bd59-6403122705c9" />

Login page

<img width="679" height="586" alt="image" src="https://github.com/user-attachments/assets/2391563d-e5b1-4930-9cd3-e8523dcdce43" />

Charts

<img width="476" height="450" alt="newplot" src="https://github.com/user-attachments/assets/9576cfd3-8699-49ab-9103-c17ede5dbe6a" />

<img width="476" height="450" alt="newplot (1)" src="https://github.com/user-attachments/assets/d87ea4e3-90cb-4886-973e-d6e08ae1c476" />

<img width="1072" height="589" alt="image" src="https://github.com/user-attachments/assets/694d9015-fab5-4c46-9978-1ab1ffaa8ba8" />

Business Insights By *AI*
<img width="1049" height="584" alt="image" src="https://github.com/user-attachments/assets/ebc22a85-8751-4015-a562-845c9cbf6a3d" />

<img width="1069" height="566" alt="image" src="https://github.com/user-attachments/assets/9ee5b516-1e53-4443-88d4-c3f31f4d8630" />

<img width="1049" height="596" alt="image" src="https://github.com/user-attachments/assets/4ffa6c9a-f77a-430e-af30-e2573500bf7f" />

<img width="1053" height="587" alt="image" src="https://github.com/user-attachments/assets/2ca3e4f0-d19c-45f4-bde8-9f568db63682" />

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
Shoping_Behaviour_Analysis/
├── templates/                 # Frontend HTML templates
│   ├── index.html             # Main Dashboard UI (Charts & Upload)
│   ├── auth.html              # Login & Registration Page
│   └── dashboard.html         # Detailed Analytics View
├── analytics_mongo.py         # Business logic for data processing & insights
├── app.py                     # Main Flask application & SocketIO server
├── config.py                  # Environment & App configuration settings
├── gemini_service.py          # Google Gemini AI integration for business insights
├── .env                       # Environment variables (API Keys, DB URIs)
├── .gitignore                 # Files and folders to be ignored by Git
├── requirements.txt           # Python dependencies (Flask, Pandas, etc.)
├── proanz_mongo_app.log       # Application runtime logs
├── uploads/                   # Temporary storage for uploaded CSV files
├── DEPLOYMENT.md              # Instructions for production deployment
├── GEMINI_SETUP.md            # Documentation for configuring Google Gemini
└── venv/                      # Python virtual environment (hidden/ignored)
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

This project is licensed under the BSL-I.O License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the application logs
- Verify MongoDB connection and data format
