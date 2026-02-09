# -*- coding: utf-8 -*-
# app.py - MongoDB version of ProAnz Analytics
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from prophet import Prophet
from datetime import timedelta, date, datetime
import numpy as np
import time
from flask_socketio import SocketIO, emit
from threading import Lock
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import logging
from logging.handlers import RotatingFileHandler
import traceback
import sys
import os
from functools import wraps
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import analytics and AI services
from analytics_mongo import generate_insights
from gemini_service import gemini_service

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proanz_mongo_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Authentication middleware
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'redirect': '/auth'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def get_current_user():
    if 'user_id' in session:
        return db.users.find_one({'_id': ObjectId(session['user_id'])})
    return None

def create_upload_session(user_id, filename, file_type):
    upload_id = str(uuid.uuid4())
    upload_session = {
        'upload_id': upload_id,
        'user_id': user_id,
        'filename': filename,
        'file_type': file_type,
        'created_at': datetime.now(),
        'status': 'processing'
    }
    db.upload_sessions.insert_one(upload_session)
    return upload_id

def update_upload_session(upload_id, status, results=None):
    update_data = {
        'status': status,
        'updated_at': datetime.now()
    }
    if results:
        update_data['results'] = results
    
    db.upload_sessions.update_one(
        {'upload_id': upload_id},
        {'$set': update_data}
    )

# MongoDB connection
# MongoDB connection
# MongoDB connection
try:
    # Debug: Print all environment variables starting with MONGO
    import os
    print("DEBUG: Environment variables:")
    for key in os.environ:
        if 'MONGO' in key or 'SECRET' in key:
            print(f"  {key} = {os.environ[key][:20]}...")
    
    MONGO_URI = os.environ.get('MONGO_URI') or os.getenv('MONGO_URI')
    
    if not MONGO_URI:
        raise ValueError("MONGO_URI not found!")
    
    client = MongoClient(MONGO_URI)
    db = client['proanz_analytics']
    logger.info('Connected to MongoDB successfully')
except Exception as e:
    logger.error(f'MongoDB connection error: {str(e)}')
    raise

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Global variables for background thread
thread = None
thread_lock = Lock()
historical_sales = []

@app.errorhandler(404)
def not_found(error):
    logger.warning(f'404 Not Found: {request.url}')
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'500 Internal Server Error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/auth')
def auth_page():
    """Render authentication page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Check if user already exists
        if db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Create user
        user = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'role': 'user',
            'created_at': datetime.now(),
            'last_login': None,
            'is_active': True
        }
        
        result = db.users.insert_one(user)
        user_id = str(result.inserted_id)
        
        # Auto-login after registration
        session['user_id'] = user_id
        session['username'] = username
        session['role'] = 'user'
        session.permanent = True
        
        logger.info(f'New user registered: {username} ({email})')
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': user_id,
                'username': username,
                'email': email,
                'role': 'user'
            }
        })
        
    except Exception as e:
        logger.error(f'Registration error: {str(e)}')
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username_or_email = data.get('username_or_email', '').strip()
        password = data.get('password', '')
        
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400
        
        # Find user
        user = db.users.find_one({
            '$or': [
                {'username': username_or_email},
                {'email': username_or_email}
            ]
        })
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.get('is_active', True):
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Update last login
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.now()}}
        )
        
        # Create session
        session['user_id'] = str(user['_id'])
        session['username'] = user['username']
        session['email'] = user['email']
        session['role'] = user.get('role', 'user')
        session.permanent = True
        
        logger.info(f'User logged in: {user["username"]}')
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'role': user.get('role', 'user')
            }
        })
        
    except Exception as e:
        logger.error(f'Login error: {str(e)}')
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    try:
        user_id = session.get('user_id')
        username = session.get('username')
        
        session.clear()
        
        if user_id:
            logger.info(f'User logged out: {username} (ID: {user_id})')
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        })
        
    except Exception as e:
        logger.error(f'Logout error: {str(e)}')
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/auth/current-user')
def current_user():
    """Get current authenticated user"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        user = get_current_user()
        if not user:
            session.clear()
            return jsonify({'error': 'User not found'}), 401
        
        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'role': user.get('role', 'user'),
                'last_login': user.get('last_login'),
                'created_at': user.get('created_at')
            }
        })
        
    except Exception as e:
        logger.error(f'Current user error: {str(e)}')
        return jsonify({'error': 'Failed to get user info'}), 500

@app.route('/api/auth/upload-history')
@login_required
def upload_history():
    """Get user's upload history"""
    try:
        user_id = session['user_id']
        
        # Get user's upload sessions
        uploads = list(db.upload_sessions.find(
            {'user_id': user_id},
            {'_id': 0, 'upload_id': 1, 'filename': 1, 'file_type': 1, 
             'status': 1, 'created_at': 1, 'updated_at': 1, 'results': 1}
        ).sort('created_at', -1))
        
        # Convert ObjectId to string and format dates
        for upload in uploads:
            upload['created_at'] = upload['created_at'].isoformat() if upload.get('created_at') else None
            upload['updated_at'] = upload['updated_at'].isoformat() if upload.get('updated_at') else None
            if '_id' in upload:
                upload['_id'] = str(upload['_id'])
            if 'user_id' in upload:
                upload['user_id'] = str(upload['user_id'])
        
        return jsonify({
            'success': True,
            'uploads': uploads
        })
        
    except Exception as e:
        logger.error(f'Upload history error: {str(e)}')
        return jsonify({'error': 'Failed to get upload history'}), 500

@app.route('/api/auth/upload-details/<upload_id>')
@login_required
def upload_details(upload_id):
    """Get detailed upload results"""
    try:
        user_id = session['user_id']
        
        # Get upload session
        upload = db.upload_sessions.find_one({
            'upload_id': upload_id,
            'user_id': user_id
        })
        
        if not upload:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Format dates and convert ObjectId to string
        upload['created_at'] = upload['created_at'].isoformat() if upload.get('created_at') else None
        upload['updated_at'] = upload['updated_at'].isoformat() if upload.get('updated_at') else None
        upload['_id'] = str(upload['_id']) if upload.get('_id') else None
        
        # Remove non-serializable fields
        if 'user_id' in upload:
            upload['user_id'] = str(upload['user_id']) if upload.get('user_id') else None
        
        return jsonify({
            'success': True,
            'upload': upload
        })
        
    except Exception as e:
        logger.error(f'Upload details error: {str(e)}')
        return jsonify({'error': 'Failed to get upload details'}), 500

@app.route('/')
def index():
    """Main dashboard - requires authentication"""
    if 'user_id' not in session:
        return redirect(url_for('auth_page'))
    try:
        logger.info('Serving index page')
        return render_template('index.html')
    except Exception as e:
        logger.error(f'Error serving index: {str(e)}')
        return jsonify({'error': 'Failed to load page'}), 500

@socketio.on('connect')
def connect():
    try:
        logger.info(f'Client connected: {request.sid}')
        emit('Connected', {'message': 'Welcome to real-time analytics!'}, to=request.sid)
    except Exception as e:
        logger.error(f'Error in connect event: {str(e)}')

@socketio.on('disconnect')
def disconnect():
    try:
        logger.info(f'Client disconnected: {request.sid}')
    except Exception as e:
        logger.error(f'Error in disconnect event: {str(e)}')

@socketio.on('start_streaming')
def start_streaming():
    global thread
    try:
        logger.info(f'Starting streaming for client: {request.sid}')
        with thread_lock:
            if thread is None:
                thread = socketio.start_background_task(background_thread)
        emit('streaming_started', {'status': 'Real-time sales data streaming initiated'})
    except Exception as e:
        logger.error(f'Error starting streaming: {str(e)}')
        emit('streaming_error', {'error': str(e)})

def background_thread():
    global historical_sales
    logger.info("Starting background thread for real-time sales simulation")
    try:
        if not historical_sales:
            last_date = date.today()
            last_value = 100
            historical_sales = [{'date': last_date.strftime('%Y-%m-%d'), 'units': last_value}]
            logger.info("Initialized default historical sales data")
        
        while True:
            try:
                last_entry = historical_sales[-1]
                trend_factor = 1.01
                change = np.random.normal(0, 10)
                new_units = max(0, last_entry['units'] * trend_factor + change)
                new_date = pd.to_datetime(last_entry['date']) + timedelta(days=1)
                
                new_data = {'date': new_date.strftime('%Y-%m-%d'), 'units': round(new_units)}
                historical_sales.append(new_data)
                
                # Store in MongoDB
                db.daily_sales.update_one(
                    {'date': new_date.date().strftime('%Y-%m-%d')},
                    {'$set': {'units_sold': new_data['units'], 'updated_at': datetime.now()}},
                    upsert=True
                )
                
                emit('new_sales_data', new_data, broadcast=True)
                logger.info(f"Emitted new sales: {new_units} on {new_date}")
                
                time.sleep(5)
            except Exception as e:
                logger.error(f'Error in background thread simulation: {str(e)}')
                time.sleep(10)
    except Exception as e:
        logger.error(f'Fatal error in background thread: {str(e)}')
        traceback.print_exc()

@app.route('/dashboard')
def dashboard():
    """Render dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard/<upload_id>')
def get_dashboard_data(upload_id):
    """Get dashboard data for specific upload session"""
    try:
        # Get user from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Find upload session
        upload_session = db.upload_sessions.find_one({
            'upload_id': upload_id,
            'user_id': user_id
        })
        
        if not upload_session:
            return jsonify({'error': 'Upload session not found'}), 404
        
        # Check if session is completed
        if upload_session.get('status') != 'completed':
            return jsonify({'error': 'Upload session not completed yet'}), 400
        
        # Return stored results
        results = upload_session.get('results', {})
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        return jsonify({'error': 'Failed to fetch dashboard data'}), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    global historical_sales
    logger.info('Received upload request')
    try:
        user_id = session['user_id']
        username = session['username']
        
        file = request.files.get('file')
        df = None
        upload_id = None
        file_type = None
        
        if file and file.filename != '':
            logger.info(f'Uploading file: {file.filename} by user: {username}')
            df = pd.read_csv(file)
            required_columns = ['product_name', 'date', 'units_sold', 'price']
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                available_columns = list(df.columns)
                error_msg = f'Missing required columns: {missing_columns}. Available columns: {available_columns}. Required format: product_name, date, units_sold, price'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            
            # Create upload session
            upload_id = create_upload_session(user_id, file.filename, 'csv')
            file_type = 'csv'
        else:
            product_names = request.form.getlist('product_name[]')
            date_strs = request.form.getlist('date[]')
            units_strs = request.form.getlist('units_sold[]')
            price_strs = request.form.getlist('price[]')
            
            logger.info(f'Manual upload by user: {username}')
            logger.info(f'Debug - Received: product_names len={len(product_names)} vals={product_names}')
            logger.info(f'Debug - Received: date_strs len={len(date_strs)} vals={date_strs}')
            logger.info(f'Debug - Received: units_strs len={len(units_strs)} vals={units_strs}')
            logger.info(f'Debug - Received: price_strs len={len(price_strs)} vals={price_strs}')
            
            if not all([product_names, date_strs, units_strs, price_strs]):
                error_msg = f'Missing required fields for manual input. Debug lens: names={len(product_names)}, dates={len(date_strs)}, units={len(units_strs)}, prices={len(price_strs)}'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            if len(product_names) != len(date_strs) or len(product_names) != len(units_strs) or len(product_names) != len(price_strs):
                error_msg = 'Mismatched number of fields across rows'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            try:
                units_sold = [float(u) for u in units_strs]
                prices = [float(p) for p in price_strs]
            except ValueError as ve:
                error_msg = f'Invalid number for units sold or price: {ve}'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            df = pd.DataFrame({
                'product_name': product_names,
                'date': pd.to_datetime(date_strs),
                'units_sold': units_sold,
                'price': prices
            })
            logger.info(f'Manual input processed: {len(df)} rows')
            
            # Create upload session
            upload_id = create_upload_session(user_id, 'Manual Entry', 'manual')
            file_type = 'manual'
        
        # Data cleaning
        logger.info('Cleaning data')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['units_sold', 'price'])

        if df.empty:
            error_msg = 'No valid data after cleaning'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        # Limit to last 1000 rows for performance
        original_len = len(df)
        if len(df) > 1000:
            df = df.tail(1000)
            logger.info(f"Data downsampled from {original_len} to last 1000 rows for performance")

        # Clear existing data in MongoDB collections
        db.sales_data.delete_many({})
        db.daily_sales.delete_many({})
        logger.info('Cleared existing MongoDB collections')

        # Insert cleaned data into MongoDB
        logger.info('Inserting data into sales_data collection')
        sales_records = []
        for index, row in df.iterrows():
            record = {
                'product_name': row['product_name'],
                'date': row['date'].date().strftime('%Y-%m-%d'),
                'units_sold': float(row['units_sold']),
                'price': float(row['price']),
                'created_at': datetime.now()
            }
            sales_records.append(record)
        
        if sales_records:
            db.sales_data.insert_many(sales_records)
            logger.info(f'Inserted {len(sales_records)} records into sales_data')

        # Aggregate product data from MongoDB
        logger.info('Aggregating product sales data from MongoDB')
        pipeline = [
            {
                '$group': {
                    '_id': '$product_name',
                    'units_sold': {'$sum': '$units_sold'},
                    'price': {'$avg': '$price'}
                }
            },
            {'$sort': {'units_sold': -1}}
        ]
        prod_sales_cursor = db.sales_data.aggregate(pipeline)
        prod_sales_data = list(prod_sales_cursor)
        
        if not prod_sales_data:
            error_msg = 'No product data found after aggregation'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400
        
        prod_sales = pd.DataFrame([
            {'product_name': item['_id'], 'units_sold': item['units_sold'], 'price': item['price']}
            for item in prod_sales_data
        ])

        # Generate analytics and insights
        insights = generate_insights(prod_sales, df)
        
        # 1. Most Selling Products
        most_selling = prod_sales.nlargest(10, 'units_sold')
        fig_most = px.bar(most_selling, x='product_name', y='units_sold', 
                          title='Most Selling Products (Top 10)', 
                          labels={'units_sold': 'Total Units Sold'})
        graph_most = fig_most.to_json()
        top_product = most_selling.iloc[0]['product_name']
        top_units = most_selling.iloc[0]['units_sold']
        note_most = f"Most trending product: {top_product} with {top_units:.0f} units sold overall."

        # 2. Low Selling Products
        low_selling = prod_sales.nsmallest(10, 'units_sold')
        fig_low = px.bar(low_selling, x='product_name', y='units_sold', 
                         title='Low Selling Products (Bottom 10)', 
                         labels={'units_sold': 'Total Units Sold'})
        graph_low = fig_low.to_json()
        low_product = low_selling.iloc[0]['product_name']
        low_units = low_selling.iloc[0]['units_sold']
        note_low = f"Lowest selling product: {low_product} with only {low_units:.0f} units sold."

        # 3. High Cost but Most Sold
        median_price = prod_sales['price'].median()
        high_price_prods = prod_sales[prod_sales['price'] > median_price].nlargest(10, 'units_sold')
        fig_high_cost = px.scatter(high_price_prods, x='price', y='units_sold', size='units_sold', 
                                   hover_name='product_name', 
                                   title='High Cost but Most Sold Products',
                                   labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_high_cost = fig_high_cost.to_json()
        if len(high_price_prods) > 0:
            top_high = high_price_prods.iloc[0]['product_name']
            note_high = f"Top high-cost seller: {top_high} at avg ${high_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_high = "No high-cost products with significant sales."

        # 4. Low Cost but Most Sold
        low_price_prods = prod_sales[prod_sales['price'] <= median_price].nlargest(10, 'units_sold')
        fig_low_cost = px.scatter(low_price_prods, x='price', y='units_sold', size='units_sold', 
                                  hover_name='product_name', 
                                  title='Low Cost but Most Sold Products',
                                  labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_low_cost = fig_low_cost.to_json()
        if len(low_price_prods) > 0:
            top_low = low_price_prods.iloc[0]['product_name']
            note_low_cost = f"Top low-cost seller: {top_low} at avg ${low_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_low_cost = "No low-cost products with significant sales."
        
        # 5. Sales Prediction with Prophet - Calculate daily_sales early for Gemini
        logger.info('Generating sales prediction with Prophet')
        daily_pipeline = [
            {
                '$group': {
                    '_id': '$date',
                    'units_sold': {'$sum': '$units_sold'}
                }
            },
            {'$sort': {'_id': 1}}
        ]
        daily_sales_cursor = db.sales_data.aggregate(daily_pipeline)
        daily_sales_data = list(daily_sales_cursor)
        
        if not daily_sales_data:
            error_msg = 'No daily sales data found for prediction'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400
        
        daily_sales = pd.DataFrame([
            {'date': item['_id'], 'units_sold': item['units_sold']}
            for item in daily_sales_data
        ])
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        
        # Generate AI-powered business insights using Gemini
        analytics_data = {
            'most_selling': most_selling.to_dict('records') if len(most_selling) > 0 else [],
            'low_selling': low_selling.to_dict('records') if len(low_selling) > 0 else [],
            'high_cost_products': high_price_prods.to_dict('records') if len(high_price_prods) > 0 else [],
            'low_cost_products': low_price_prods.to_dict('records') if len(low_price_prods) > 0 else [],
            'daily_sales': daily_sales.to_dict('records') if len(daily_sales) > 0 else [],
            'product_performance': prod_sales.to_dict('records') if len(prod_sales) > 0 else [],
            'insights': insights
        }
        
        ai_insights = gemini_service.generate_business_insights(analytics_data)
        
        # Ensure insights are JSON serializable
        if isinstance(insights, dict):
            # Convert any pandas objects to JSON-serializable format
            for key, value in insights.items():
                if hasattr(value, 'to_dict'):
                    insights[key] = value.to_dict()
                elif hasattr(value, 'tolist'):
                    insights[key] = value.tolist()
                elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                    insights[key] = list(value)

        # 1. Most Selling Products
        most_selling = prod_sales.nlargest(10, 'units_sold')
        fig_most = px.bar(most_selling, x='product_name', y='units_sold', 
                          title='Most Selling Products (Top 10)', 
                          labels={'units_sold': 'Total Units Sold'})
        graph_most = fig_most.to_json()
        top_product = most_selling.iloc[0]['product_name']
        top_units = most_selling.iloc[0]['units_sold']
        note_most = f"Most trending product: {top_product} with {top_units:.0f} units sold overall."

        # 2. Low Selling Products
        low_selling = prod_sales.nsmallest(10, 'units_sold')
        fig_low = px.bar(low_selling, x='product_name', y='units_sold', 
                         title='Low Selling Products (Bottom 10)', 
                         labels={'units_sold': 'Total Units Sold'})
        graph_low = fig_low.to_json()
        low_product = low_selling.iloc[0]['product_name']
        low_units = low_selling.iloc[0]['units_sold']
        note_low = f"Lowest selling product: {low_product} with only {low_units:.0f} units sold."

        # 3. High Cost but Most Sold
        median_price = prod_sales['price'].median()
        high_price_prods = prod_sales[prod_sales['price'] > median_price].nlargest(10, 'units_sold')
        fig_high_cost = px.scatter(high_price_prods, x='price', y='units_sold', size='units_sold', 
                                   hover_name='product_name', 
                                   title='High Cost but Most Sold Products',
                                   labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_high_cost = fig_high_cost.to_json()
        if len(high_price_prods) > 0:
            top_high = high_price_prods.iloc[0]['product_name']
            note_high = f"Top high-cost seller: {top_high} at avg ${high_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_high = "No high-cost products with significant sales."

        # 4. Low Cost but Most Sold
        low_price_prods = prod_sales[prod_sales['price'] <= median_price].nlargest(10, 'units_sold')
        fig_low_cost = px.scatter(low_price_prods, x='price', y='units_sold', size='units_sold', 
                                  hover_name='product_name', 
                                  title='Low Cost but Most Sold Products',
                                  labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_low_cost = fig_low_cost.to_json()
        if len(low_price_prods) > 0:
            top_low = low_price_prods.iloc[0]['product_name']
            note_low_cost = f"Top low-cost seller: {top_low} at avg ${low_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_low_cost = "No low-cost products with significant sales."

        # 5. Sales Prediction with Prophet - Continue with existing daily_sales
        logger.info('Generating sales prediction with Prophet')
        
        if len(daily_sales) < 2:
            # Insert single entry into daily_sales collection
            if len(daily_sales) == 1:
                row = daily_sales.iloc[0]
                db.daily_sales.update_one(
                    {'date': row['date'].date().strftime('%Y-%m-%d')},
                    {'$set': {'units_sold': float(row['units_sold']), 'updated_at': datetime.now()}},
                    upsert=True
                )
                logger.info('Inserted single daily sales into DailySales collection')
            
            # Handle single entry: just plot the point, no forecast
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['units_sold'], 
                                          mode='markers', name='Actual Sales', 
                                          marker=dict(color='blue', size=20)))
            fig_pred.update_layout(title='Sales Data (Single Entry - No Prediction Available)', 
                                   xaxis_title='Date', yaxis_title='Units Sold')
            graph_pred = fig_pred.to_json()
            note_pred = "Single data point recorded. Provide more dates via CSV for full prediction analysis with Prophet model."
        else:
            
            prophet_df = daily_sales.rename(columns={'date': 'ds', 'units_sold': 'y'})
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['units_sold'], 
                                          mode='lines+markers', name='Actual Sales', line=dict(color='blue')))
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                          mode='lines', name='Predicted Sales', line=dict(color='red', dash='dot')))
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                          fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                          fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', fillcolor='rgba(255,0,0,0.2)', name='Prediction Interval'))
            fig_pred.update_layout(title='Advanced Sales Prediction (Prophet Model, Next 30 Days)', 
                                   xaxis_title='Date', yaxis_title='Units Sold')
            graph_pred = fig_pred.to_json()
            avg_future = forecast['yhat'].tail(30).mean()
            note_pred = f"Predicted average daily sales for next 30 days: {avg_future:.0f} units. Prophet model fitted successfully."

        # Sync in-memory historical sales with MongoDB for streaming
        daily_cursor = db.daily_sales.find().sort('date', 1)
        historical_sales = [{'date': doc['date'], 'units': float(doc['units_sold'])} 
                           for doc in daily_cursor]
        logger.info('Updated historical sales from MongoDB for streaming')

        # 6. Product-Wise Detailed Report
        logger.info('Generating product-wise report')
        fig_report = go.Figure()
        fig_report.add_trace(go.Bar(x=prod_sales['product_name'], y=prod_sales['units_sold'], 
                                    name='Total Units Sold', marker_color='lightblue'))
        fig_report.add_trace(go.Scatter(x=prod_sales['product_name'], y=prod_sales['price'] * prod_sales['units_sold'], 
                                        mode='markers+lines', name='Revenue', yaxis='y2', line=dict(color='green')))
        fig_report.update_layout(title='Product-Wise Detailed Report (Bar: Sales, Line: Revenue)',
                                 xaxis_title='Product Name',
                                 yaxis=dict(title='Units Sold'),
                                 yaxis2=dict(title='Revenue', overlaying='y', side='right'))
        graph_report = fig_report.to_json()
        total_sales = prod_sales['units_sold'].sum()
        avg_sales = prod_sales['units_sold'].mean()
        note_report = f"Total sales across all products: {total_sales:.0f} units. Average per product: {avg_sales:.0f} units."

        logger.info('Upload and analysis completed successfully')
        
        # Convert all graph data to JSON strings to ensure serialization
        response_data = {
            'most_selling': {'graph': graph_most, 'note': note_most},
            'low_selling': {'graph': graph_low, 'note': note_low},
            'high_cost_high_sales': {'graph': graph_high_cost, 'note': note_high},
            'low_cost_high_sales': {'graph': graph_low_cost, 'note': note_low_cost},
            'sales_prediction': {'graph': graph_pred, 'note': note_pred},
            'product_report': {'graph': graph_report, 'note': note_report},
            'insights': insights,
            'ai_insights': ai_insights,
            'upload_id': upload_id,
            'file_type': file_type
        }
        
        # Update upload session with results
        if upload_id:
            update_upload_session(upload_id, 'completed', response_data)
        
        return jsonify(response_data)

    except pd.errors.EmptyDataError as e:
        error_msg = 'Empty CSV file or invalid format'
        logger.error(f'Pandas error in upload: {str(e)}')
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f'Unexpected error during upload: {str(e)}'
        logger.error(f'Unexpected error in upload: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    logger.info('MongoDB-based ProAnz app starting...')
    socketio.run(app, debug=True)
