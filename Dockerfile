# Use the official Python lightweight image
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for pdf2image (poppler) and sqlite
RUN apt-get update && apt-get install -y \
    poppler-utils \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure data directories exist for SQLite and ChromaDB
RUN mkdir -p data/approved_invoices

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501
