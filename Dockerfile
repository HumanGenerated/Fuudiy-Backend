FROM python:3.9-dev

# Avoid interactive prompts
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
    libc6 \
    libsnappy1v5 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Copy and install Python dependencies with PyTorch CPU wheel
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

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