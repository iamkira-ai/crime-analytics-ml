# Crime Hotspot Prediction System

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-powered system for predicting crime hotspots using historical crime incident data. Built with Python, FastAPI, scikit-learn, and Docker, featuring a complete CI/CD pipeline with GitHub Actions.

## Features

- **Machine Learning Model**: Random Forest classifier for crime risk prediction
- **Geospatial Analysis**: DBSCAN clustering for hotspot identification
- **REST API**: Fast and scalable API built with FastAPI
- **Docker Support**: Fully containerized application
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Comprehensive Testing**: Unit tests, integration tests, and API tests
- **Model Persistence**: Save and load trained models
- **Real-time Predictions**: Get crime risk levels for any location and time

## Architecture

```
├── app.py                 # Main application with ML model and API
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
├── requirements.txt      # Python dependencies
├── tests/                # Test suite
│   └── test_app.py       # Application tests
├── .github/              # GitHub Actions workflows
│   └── workflows/
│       └── ci-cd.yml     # CI/CD pipeline
├── data/                 # Data directory (mount point)
├── models/               # Trained model storage
└── logs/                 # Application logs
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Git

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamkira-ai/crime-analytics-ml.git
   cd crime-hotspot-prediction
   ```

2. **Place your crime data**
   ```bash
   mkdir -p data
   # Copy your Crime_Incidents_in_2024.csv to the data/ directory
   cp /path/to/Crime_Incidents_in_2024.csv data/
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Local Development

1. **Setup virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python app.py train data/Crime_Incidents_in_2024.csv
   ```

4. **Run the API**
   ```bash
   python app.py
   ```

## Usage

### Training a Model

```bash
# Using Docker
docker-compose exec crime-predictor python app.py train data/Crime_Incidents_in_2024.csv

# Local development
python app.py train data/Crime_Incidents_in_2024.csv
```

### Making Predictions

#### Via API

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 38.9072,
       "longitude": -77.0369,
       "hour": 14,
       "day_of_week": 1,
       "month": 6,
       "shift": "DAY",
       "district": 1,
       "ward": 1
     }'
```

#### Response Format

```json
{
  "risk_level": 1,
  "risk_probability": [0.3, 0.5, 0.2],
  "risk_description": "Medium"
}
```

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Crime risk prediction
- `GET /docs` - Interactive API documentation

## Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

### Run Tests in Docker

```bash
docker-compose exec crime-predictor pytest tests/ -v
```

## Configuration

### Environment Variables

- `PYTHONPATH`: Application path (default: `/app`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Model Parameters

The Random Forest model uses these default parameters:
- `n_estimators`: 100
- `max_depth`: 10
- `random_state`: 42

You can modify these in the `train_model` method of the `CrimePredictor` class.

## Model Details

### Features Used

- **Temporal Features**: Hour, day of week, month, weekend indicator
- **Geospatial Features**: Latitude, longitude
- **Categorical Features**: Shift, method, offense type, district, ward
- **Risk Levels**: 0 (Low), 1 (Medium), 2 (High)

### Risk Level Calculation

Risk levels are determined using DBSCAN clustering:
- **High Risk (2)**: Areas with >50 crime incidents in cluster
- **Medium Risk (1)**: Areas with 20-50 crime incidents in cluster  
- **Low Risk (0)**: Areas with <20 crime incidents or noise points

## Deployment

### GitHub Actions CI/CD

The project includes a comprehensive CI/CD pipeline that:

1. **Testing**: Runs tests on multiple Python versions
2. **Security**: Vulnerability scanning with Trivy
3. **Quality**: Code linting and formatting checks
4. **Building**: Multi-platform Docker image builds
5. **Deployment**: Automated deployment to staging/production

### Production Deployment

1. **Setup production environment**
   ```bash
   docker-compose --profile production up -d
   ```

2. **With monitoring**
   ```bash
   docker-compose --profile monitoring up -d
   ```

3. **Access monitoring**
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## API Documentation

### Prediction Request Schema

```python
{
  "latitude": float,      # Latitude coordinate
  "longitude": float,     # Longitude coordinate  
  "hour": int,           # Hour of day (0-23)
  "day_of_week": int,    # Day of week (0-6, Monday=0)
  "month": int,          # Month (1-12)
  "shift": str,          # "DAY", "EVENING", or "MIDNIGHT"
  "district": int,       # Police district number
  "ward": int            # Ward number
}
```

### Response Schema

```python
{
  "risk_level": int,              # 0, 1, or 2
  "risk_probability": [float],    # Probability for each risk level
  "risk_description": str         # "Low", "Medium", or "High"
}
```