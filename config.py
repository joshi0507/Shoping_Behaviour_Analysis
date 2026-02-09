import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/proanz_analytics')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Environment settings
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('FLASK_ENV', 'development')
    
    # Production settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

class ProductionConfig(Config):
    DEBUG = False
    
class DevelopmentConfig(Config):
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
