FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9, pip, and build essentials
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    unixodbc-dev \
    unixodbc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Copy and install Python dependencies with PyTorch CPU wheel
COPY requirements.txt .

# Update pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set working directory and copy project
WORKDIR /app/app
COPY ./app /app/app


ENV PYTHONPATH=/app
ENV PORT=8000

# Expose FastAPI port
EXPOSE 8000

# COPY gcs-key.json gcs-key.json

# ENV LOG4J_DISABLE_JMX=true

# Default start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]