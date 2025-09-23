# Kairos ADO - Dockerfile
# Defines the container image for the Kairos application.

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /kairos

# Install system dependencies that might be needed
# (e.g., for psycopg2, prophet, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# (The CMD will be set in the docker-compose file to allow for flexibility)

