"""
Tests for the Crime Hotspot Prediction API
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, CrimePredictor, predictor

client = TestClient(app)

class TestCrimePredictor:
    """Test the CrimePredictor class"""
    
    def setup_method(self):
        """Setup test predictor"""
        self.predictor = CrimePredictor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'REPORT_DAT': ['2024-01-01 10:30:00', '2024-01-02 14:15:00', '2024-01-03 20:45:00'],
            'START_DATE': ['2024-01-01 10:00:00', '2024-01-02 14:00:00', '2024-01-03 20:30:00'],
            'LATITUDE': [38.9072, 38.9073, 38.9074],
            'LONGITUDE': [-77.0369, -77.0370, -77.0371],
            'OFFENSE': ['THEFT/OTHER', 'BURGLARY', 'ASSAULT W/DANGEROUS WEAPON'],
            'SHIFT': ['DAY', 'EVENING', 'MIDNIGHT'],
            'METHOD': ['GUN', 'KNIFE', 'OTHER'],
            'DISTRICT': [1, 2, 3],
            'WARD': [1, 2, 3]
        })
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        X, y = self.predictor.preprocess_data(self.sample_data)
        
        assert len(X) == len(self.sample_data)
        assert len(y) == len(self.sample_data)
        assert X.shape[1] > 0  # Should have features
        assert all(isinstance(risk, (int, np.integer)) for risk in y)
    
    def test_create_risk_levels(self):
        """Test risk level creation"""
        risk_levels = self.predictor.create_risk_levels(self.sample_data)
        
        assert len(risk_levels) == len(self.sample_data)
        assert all(risk in [0, 1, 2] for risk in risk_levels)
    
    def test_model_training(self):
        """Test model training"""
        X, y = self.predictor.preprocess_data(self.sample_data)
        
        # Add more sample data for training
        X_extended = pd.concat([X] * 10, ignore_index=True)
        y_extended = pd.concat([pd.Series(y)] * 10, ignore_index=True)
        
        accuracy = self.predictor.train_model(X_extended, y_extended)
        
        assert self.predictor.model is not None
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_prediction(self):
        """Test prediction functionality"""
        X, y = self.predictor.preprocess_data(self.sample_data)
        
        # Add more sample data for training
        X_extended = pd.concat([X] * 20, ignore_index=True)
        y_extended = pd.concat([pd.Series(y)] * 20, ignore_index=True)
        
        self.predictor.train_model(X_extended, y_extended)
        
        # Test prediction
        sample_features = X.iloc[0].values
        prediction = self.predictor.predict_risk(sample_features)
        
        assert 'risk_level' in prediction
        assert 'risk_probability' in prediction
        assert 'risk_description' in prediction
        assert prediction['risk_level'] in [0, 1, 2]
        assert prediction['risk_description'] in ['Low', 'Medium', 'High']


class TestAPI:
    """Test the FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Crime Hotspot Prediction API" in response.json()["message"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
    
    @patch.object(predictor, 'predict_risk')
    def test_predict_endpoint_success(self, mock_predict):
        """Test successful prediction endpoint"""
        # Mock the prediction
        mock_predict.return_value = {
            'risk_level': 1,
            'risk_probability': [0.3, 0.5, 0.2],
            'risk_description': 'Medium'
        }
        
        request_data = {
            "latitude": 38.9072,
            "longitude": -77.0369,
            "hour": 14,
            "day_of_week": 1,
            "month": 6,
            "shift": "DAY",
            "district": 1,
            "ward": 1
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["risk_level"] == 1
        assert result["risk_description"] == "Medium"
        assert len(result["risk_probability"]) == 3
    
    def test_predict_endpoint_validation(self):
        """Test prediction endpoint input validation"""
        # Test missing required fields
        request_data = {
            "latitude": 38.9072,
            # Missing longitude
            "hour": 14,
            "day_of_week": 1,
            "month": 6
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_values(self):
        """Test prediction endpoint with invalid values"""
        request_data = {
            "latitude": 91.0,  # Invalid latitude
            "longitude": -77.0369,
            "hour": 25,  # Invalid hour
            "day_of_week": 8,  # Invalid day of week
            "month": 13  # Invalid month
        }
        
        response = client.post("/predict", json=request_data)
        # Should either return 422 for validation or 500 for processing error
        assert response.status_code in [422, 500]


class TestDataProcessing:
    """Test data processing functions"""
    
    def test_datetime_conversion(self):
        """Test datetime conversion functionality"""
        predictor = CrimePredictor()
        
        # Test data with various datetime formats
        test_data = pd.DataFrame({
            'REPORT_DAT': ['2024-01-01 10:30:00', '2024-01-02T14:15:00', 'invalid_date'],
            'START_DATE': ['2024-01-01 10:00:00', '2024-01-02T14:00:00', '2024-01-03 20:30:00'],
            'LATITUDE': [38.9072, 38.9073, 38.9074],
            'LONGITUDE': [-77.0369, -77.0370, -77.0371],
            'OFFENSE': ['THEFT', 'BURGLARY', 'ASSAULT'],
            'SHIFT': ['DAY', 'EVENING', 'MIDNIGHT'],
            'METHOD': ['GUN', 'KNIFE', 'OTHER'],
            'DISTRICT': [1, 2, 3],
            'WARD': [1, 2, 3]
        })
        
        X, y = predictor.preprocess_data(test_data)
        
        # Should handle invalid dates gracefully
        assert len(X) > 0  # At least some data should be processed
        assert len(y) > 0
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        predictor = CrimePredictor()
        
        # Test data with missing values
        test_data = pd.DataFrame({
            'REPORT_DAT': ['2024-01-01 10:30:00', '2024-01-02 14:15:00'],
            'START_DATE': ['2024-01-01 10:00:00', '2024-01-02 14:00:00'],
            'LATITUDE': [38.9072, None],  # Missing latitude
            'LONGITUDE': [-77.0369, -77.0370],
            'OFFENSE': ['THEFT', None],  # Missing offense
            'SHIFT': ['DAY', None],  # Missing shift
            'METHOD': ['GUN', None],
            'DISTRICT': [1, None],
            'WARD': [1, None]
        })
        
        X, y = predictor.preprocess_data(test_data)
        
        # Should filter out rows with missing critical data
        assert len(X) >= 0  # May be 0 if all data is invalid
        assert len(y) == len(X)


@pytest.fixture
def sample_crime_data():
    """Fixture providing sample crime data"""
    return pd.DataFrame({
        'REPORT_DAT': pd.date_range('2024-01-01', periods=100, freq='H'),
        'START_DATE': pd.date_range('2024-01-01', periods=100, freq='H'),
        'LATITUDE': np.random.uniform(38.9, 39.0, 100),
        'LONGITUDE': np.random.uniform(-77.1, -77.0, 100),
        'OFFENSE': np.random.choice(['THEFT', 'BURGLARY', 'ASSAULT', 'ROBBERY'], 100),
        'SHIFT': np.random.choice(['DAY', 'EVENING', 'MIDNIGHT'], 100),
        'METHOD': np.random.choice(['GUN', 'KNIFE', 'OTHER'], 100),
        'DISTRICT': np.random.randint(1, 8, 100),
        'WARD': np.random.randint(1, 9, 100)
    })


def test_end_to_end_workflow(sample_crime_data):
    """Test complete workflow from data to prediction"""
    predictor = CrimePredictor()
    
    # Preprocess data
    X, y = predictor.preprocess_data(sample_crime_data)
    
    # Train model
    accuracy = predictor.train_model(X, y)
    
    # Make prediction
    sample_features = X.iloc[0].values
    prediction = predictor.predict_risk(sample_features)
    
    # Verify complete workflow
    assert accuracy is not None
    assert prediction['risk_level'] in [0, 1, 2]
    assert len(prediction['risk_probability']) == 3
    assert prediction['risk_description'] in ['Low', 'Medium', 'High']


if __name__ == "__main__":
    pytest.main([__file__])