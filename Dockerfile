# Use official Python image as base
FROM python:3.11-slim

# Install system dependencies for numpy, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Set the default command to run your script
CMD ["python", "Feature_extraction.py"]