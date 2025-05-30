FROM python:3.10-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
