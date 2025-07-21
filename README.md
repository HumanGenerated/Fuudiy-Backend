# Fuudiy Backend

This is the backend service for the Fuudiy application, built with FastAPI and MongoDB.

## Features

- RESTful API built with FastAPI
- MongoDB database integration
- Google Cloud Storage for file storage
- JWT-based authentication
- PySpark integration for data processing
- Docker containerization support

## Prerequisites

- Python 3.9+
- MongoDB
- Google Cloud Storage account (for file storage)
- Java 11 (required for PySpark)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Fuudiy-Backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
MONGODB_URL=your_mongodb_url
JWT_SECRET=your_jwt_secret
GCS_BUCKET_NAME=your_gcs_bucket_name
```

5. Place your Google Cloud Storage key file (`gcs-key.json`) in the root directory.

## Running the Application

### Development Mode

```bash
uvicorn app.app:app --reload
```

The API will be available at `http://localhost:8000`

### Using Docker

1. Build the Docker image:
```bash
docker build -t fuudiy-backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 fuudiy-backend
```

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Testing

Run tests using pytest:
```bash
pytest
```

## Project Structure

```
Fuudiy-Backend/
├── app/                 # Main application code
├── tests/              # Test files
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .dockerignore      # Docker ignore file
├── .gitignore         # Git ignore file
└── gcs-key.json       # Google Cloud Storage credentials
```

## Dependencies

Key dependencies include:
- FastAPI: Web framework
- Motor: Async MongoDB driver
- PySpark: Data processing
- Google Cloud Storage: File storage
- PyJWT: JWT authentication
- Pydantic: Data validation
- Uvicorn: ASGI server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
