# Crime Hotspot Prediction System - Makefile
# Convenient commands for development and deployment

.PHONY: help install test train run docker-build docker-run clean lint format

# Default target
help:
	@echo "Crime Hotspot Prediction System - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install         Install dependencies in virtual environment"
	@echo "  setup-docker    Setup with Docker"
	@echo "  setup-local     Setup for local development"
	@echo ""
	@echo "Development Commands:"
	@echo "  test            Run all tests"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and isort"
	@echo "  train           Train the ML model with sample data"
	@echo "  run             Run the API locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run with Docker Compose"
	@echo "  docker-test     Run tests in Docker"
	@echo "  docker-stop     Stop Docker services"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean           Clean up temporary files"
	@echo "  sample-data     Create sample data file"
	@echo "  logs            View application logs"

# Setup commands
install:
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Dependencies installed. Activate with: source venv/bin/activate"

setup-docker:
	python setup.py --docker --sample-data

setup-local:
	python setup.py --local --sample-data

# Development commands
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/index.html"

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black .
	isort .
	@echo "✅ Code formatted with black and isort"

train:
	@if [ ! -f "data/Crime_Incidents_in_2024.csv" ]; then \
		echo "⚠️  No data file found. Creating sample data..."; \
		make sample-data; \
	fi
	python app.py train data/Crime_Incidents_in_2024.csv
	@echo "✅ Model training complete"

run:
	@if [ ! -f "models/crime_predictor.joblib" ]; then \
		echo "⚠️  No trained model found. Training model first..."; \
		make train; \
	fi
	python app.py
	@echo "🚀 API running at http://localhost:8000"

# Docker commands
docker-build:
	docker build -t crime-predictor .
	@echo "✅ Docker image built"

docker-run:
	docker-compose up --build -d
	@echo "🚀 Services running:"
	@echo "  API: http://localhost:8000"
	@echo "  Docs: http://localhost:8000/docs"

docker-test:
	docker-compose exec crime-predictor python -m pytest tests/ -v

docker-stop:
	docker-compose down
	@echo "🛑 Docker services stopped"

docker-logs:
	docker-compose logs -f crime-predictor

# Utility commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	@echo "🧹 Cleaned up temporary files"

sample-data:
	mkdir -p data
	@echo "Creating sample data file..."
	@echo "X,Y,CCN,REPORT_DAT,SHIFT,METHOD,OFFENSE,BLOCK,XBLOCK,YBLOCK,WARD,ANC,DISTRICT,PSA,NEIGHBORHOOD_CLUSTER,BLOCK_GROUP,CENSUS_TRACT,VOTING_PRECINCT,LATITUDE,LONGITUDE,BID,START_DATE,END_DATE,OBJECTID,OCTO_RECORD_ID" > data/Crime_Incidents_in_2024.csv
	@echo "-77.0369,38.9072,24001001,2024-01-01 10:30:00,DAY,GUN,THEFT/OTHER,1000 BLOCK OF MAIN ST,1000,2000,1,1A01,1,101,Cluster_1,001,100.01,Precinct_1,38.9072,-77.0369,BID_1,2024-01-01 10:00:00,2024-01-01 11:00:00,1,REC_001" >> data/Crime_Incidents_in_2024.csv
	@echo "-77.0370,38.9073,24001002,2024-01-01 14:15:00,EVENING,KNIFE,BURGLARY,1100 BLOCK OF MAIN ST,1100,2100,2,2A01,2,201,Cluster_2,002,100.02,Precinct_2,38.9073,-77.0370,BID_2,2024-01-01 14:00:00,2024-01-01 15:00:00,2,REC_002" >> data/Crime_Incidents_in_2024.csv
	@echo "-77.0371,38.9074,24001003,2024-01-01 20:45:00,MIDNIGHT,OTHER,ASSAULT W/DANGEROUS WEAPON,1200 BLOCK OF MAIN ST,1200,2200,3,3A01,3,301,Cluster_3,003,100.03,Precinct_3,38.9074,-77.0371,BID_3,2024-01-01 20:30:00,2024-01-01 21:30:00,3,REC_003" >> data/Crime_Incidents_in_2024.csv
	@echo "📊 Sample data created at data/Crime_Incidents_in_2024.csv"

logs:
	@if [ -f "logs/app.log" ]; then \
		tail -f logs/app.log; \
	else \
		echo "No log files found"; \
	fi

# Development workflow shortcuts
dev: setup-local train run

docker-dev: setup-docker docker-run

# Quick quality check
check: lint test
	@echo "✅ Quality checks passed"

# Full CI simulation
ci: clean format lint test train
	@echo "✅ CI simulation complete"

# Production deployment
deploy: docker-build
	@echo "🚀 Ready for deployment"
	@echo "Run: docker-compose --profile production up -d"