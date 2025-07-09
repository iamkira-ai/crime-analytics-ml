"""
Crime Hotspot Prediction System
A machine learning application for predicting crime risk levels based on historical data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import DBSCAN
import joblib
import logging
from datetime import datetime
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Preprocess crime incident data for ML training"""
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Convert date columns
        data['REPORT_DAT'] = pd.to_datetime(data['REPORT_DAT'], errors='coerce')
        data['START_DATE'] = pd.to_datetime(data['START_DATE'], errors='coerce')
        
        # Extract time features
        data['hour'] = data['REPORT_DAT'].dt.hour
        data['day_of_week'] = data['REPORT_DAT'].dt.dayofweek
        data['month'] = data['REPORT_DAT'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Handle missing values
        data = data.dropna(subset=['LATITUDE', 'LONGITUDE', 'OFFENSE'])
        
        # Create risk levels based on crime density
        risk_levels = self.create_risk_levels(data)
        data['risk_level'] = risk_levels
        
        # Select and encode features
        categorical_features = ['SHIFT', 'METHOD', 'OFFENSE', 'DISTRICT', 'WARD']
        numerical_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'LATITUDE', 'LONGITUDE']
        
        # Encode categorical variables
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[feature] = data[feature].fillna('Unknown')
                data[feature + '_encoded'] = le.fit_transform(data[feature])
                self.label_encoders[feature] = le
        
        # Select final features
        encoded_categorical = [f + '_encoded' for f in categorical_features if f in data.columns]
        self.feature_columns = numerical_features + encoded_categorical
        
        X = data[self.feature_columns].fillna(0)
        y = data['risk_level']
        
        logger.info(f"Preprocessing complete. Features: {len(self.feature_columns)}, Samples: {len(X)}")
        return X, y
    
    def create_risk_levels(self, data):
        """Create risk levels based on spatial clustering of crimes"""
        # Use DBSCAN to identify crime hotspots
        coords = data[['LATITUDE', 'LONGITUDE']].dropna()
        
        # Normalize coordinates for clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.1, min_samples=10)
        clusters = dbscan.fit_predict(coords_scaled)
        
        # Calculate crime density for each point
        risk_levels = np.zeros(len(data))
        valid_indices = data[['LATITUDE', 'LONGITUDE']].dropna().index
        
        for i, (idx, cluster) in enumerate(zip(valid_indices, clusters)):
            if cluster == -1:  # Noise points
                risk_levels[idx] = 0  # Low risk
            else:
                # Count crimes in the same cluster
                cluster_size = np.sum(clusters == cluster)
                if cluster_size > 50:
                    risk_levels[idx] = 2  # High risk
                elif cluster_size > 20:
                    risk_levels[idx] = 1  # Medium risk
                else:
                    risk_levels[idx] = 0  # Low risk
        
        return risk_levels
    
    def train_model(self, X, y):
        """Train the crime prediction model"""
        logger.info("Training crime prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_risk(self, features):
        """Predict crime risk level for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        risk_level = self.model.predict(features_scaled)[0]
        risk_probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'risk_level': int(risk_level),
            'risk_probability': risk_probability.tolist(),
            'risk_description': ['Low', 'Medium', 'High'][int(risk_level)]
        }
    
    def save_model(self, filepath='models/crime_predictor.joblib'):
        """Save the trained model and preprocessing objects"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/crime_predictor.joblib'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")

# FastAPI Application
app = FastAPI(title="Crime Hotspot Prediction API", version="1.0.0")

# Global predictor instance
predictor = CrimePredictor()

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    hour: int
    day_of_week: int
    month: int
    shift: Optional[str] = "DAY"
    district: Optional[int] = 1
    ward: Optional[int] = 1

class PredictionResponse(BaseModel):
    risk_level: int
    risk_probability: List[float]
    risk_description: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    predictor.load_model()

@app.get("/")
async def root():
    return {"message": "Crime Hotspot Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_crime_risk(request: PredictionRequest):
    """Predict crime risk level for given location and time"""
    try:
        # Prepare features (order must match training)
        features = [
            request.hour,
            request.day_of_week,
            request.month,
            1 if request.day_of_week >= 5 else 0,  # is_weekend
            request.latitude,
            request.longitude,
            0,  # SHIFT_encoded (simplified)
            0,  # METHOD_encoded
            0,  # OFFENSE_encoded
            request.district if request.district else 1,  # DISTRICT_encoded
            request.ward if request.ward else 1,  # WARD_encoded
        ]
        
        # Pad with zeros if needed to match expected feature count
        while len(features) < len(predictor.feature_columns):
            features.append(0)
        
        prediction = predictor.predict_risk(features)
        return PredictionResponse(**prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def train_model_from_csv(csv_path: str):
    """Train model from CSV file"""
    logger.info(f"Loading data from {csv_path}")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Preprocess and train
        X, y = predictor.preprocess_data(df)
        accuracy = predictor.train_model(X, y)
        
        # Save model
        predictor.save_model()
        
        logger.info(f"Training completed with accuracy: {accuracy:.3f}")
        return accuracy
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        csv_file = sys.argv[2] if len(sys.argv) > 2 else "data/Crime_Incidents_in_2024.csv"
        train_model_from_csv(csv_file)
    else:
        # API mode
        uvicorn.run(app, host="0.0.0.0", port=8000)