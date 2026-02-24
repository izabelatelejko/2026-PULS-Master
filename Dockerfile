# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Azure Storage SDK
RUN pip install --no-cache-dir azure-storage-blob

# Copy the entire project
COPY . .

# Install the project in development mode
RUN pip install -e .

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - runs the entrypoint script
ENTRYPOINT ["python", "/app/azure_run.py"]
