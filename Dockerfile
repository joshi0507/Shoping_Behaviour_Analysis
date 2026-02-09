FROM python:3.11-slim

# Install system dependencies required for Prophet / Stan
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    git \
    curl \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*


# Create workdir
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .



# Start your app using Render's $PORT
CMD gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app

