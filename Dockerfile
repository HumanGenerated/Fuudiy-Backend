# Base image with Java 11 (needed for PySpark)
FROM openjdk:11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9, pip, and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    gcc \
    g++ \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Copy and install Python dependencies with PyTorch CPU wheel
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# Set working directory and copy project
WORKDIR /app
COPY . .

ENV PYTHONPATH=/app
ENV PORT=8000

# Expose FastAPI port
EXPOSE 8000

# Start the app (make sure your FastAPI instance is in app.py as `app`)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
