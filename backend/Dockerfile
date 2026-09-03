FROM python:3.11-slim

# Install system dependencies for PDF rendering and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend codebase
COPY . .

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose container port
EXPOSE 8000

# Run uvicorn server dynamically on $PORT provided by Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
