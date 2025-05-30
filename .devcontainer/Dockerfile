# Base image
FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git build-essential curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install --no-cache-dir --upgrade pip

# Add any additional Python dependencies here
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

# Optional: copy project files
# COPY . .

CMD [ "bash" ]
