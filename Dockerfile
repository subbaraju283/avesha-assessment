# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements-frozen.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-frozen.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/documents data/kb data/rag logs

# Set default command to bash for interactive use
CMD ["/bin/bash"] 