#!/usr/bin/env python3
"""
Setup script for Crime Hotspot Prediction System
Helps users get started with the project quickly
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs', 'tests']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def setup_virtual_environment():
    """Setup Python virtual environment"""
    if not Path('venv').exists():
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print("Installing dependencies...")
    run_command(f"{pip_path} install --upgrade pip")
    run_command(f"{pip_path} install -r requirements.txt")

def setup_docker():
    """Setup Docker environment"""
    if not run_command("docker --version", check=False):
        print("Docker not found. Please install Docker first.")
        return False
    
    print("Building Docker image...")
    run_command("docker build -t crime-predictor .")
    return True

def download_sample_data():
    """Download or create sample data"""
    data_file = Path("data/Crime_Incidents_in_2024.csv")
    if data_file.exists():
        print(f"Data file already exists: {data_file}")
        return
    
    print("Creating sample data file...")
    # Create a minimal sample CSV for testing
    sample_data = """X,Y,CCN,REPORT_DAT,SHIFT,METHOD,OFFENSE,BLOCK,XBLOCK,YBLOCK,WARD,ANC,DISTRICT,PSA,NEIGHBORHOOD_CLUSTER,BLOCK_GROUP,CENSUS_TRACT,VOTING_PRECINCT,LATITUDE,LONGITUDE,BID,START_DATE,END_DATE,OBJECTID,OCTO_RECORD_ID
-77.0369,38.9072,24001001,2024-01-01 10:30:00,DAY,GUN,THEFT/OTHER,1000 BLOCK OF MAIN ST,1000,2000,1,1A01,1,101,Cluster_1,001,100.01,Precinct_1,38.9072,-77.0369,BID_1,2024-01-01 10:00:00,2024-01-01 11:00:00,1,REC_001
-77.0370,38.9073,24001002,2024-01-01 14:15:00,EVENING,KNIFE,BURGLARY,1100 BLOCK OF MAIN ST,1100,2100,2,2A01,2,201,Cluster_2,002,100.02,Precinct_2,38.9073,-77.0370,BID_2,2024-01-01 14:00:00,2024-01-01 15:00:00,2,REC_002
-77.0371,38.9074,24001003,2024-01-01 20:45:00,MIDNIGHT,OTHER,ASSAULT W/DANGEROUS WEAPON,1200 BLOCK OF MAIN ST,1200,2200,3,3A01,3,301,Cluster_3,003,100.03,Precinct_3,38.9074,-77.0371,BID_3,2024-01-01 20:30:00,2024-01-01 21:30:00,3,REC_003"""
    
    with open(data_file, 'w') as f:
        f.write(sample_data)
    
    print(f"Created sample data file: {data_file}")
    print("Note: Replace this with your actual crime data for production use.")

def run_tests():
    """Run the test suite"""
    print("Running tests...")
    run_command("python -m pytest tests/ -v")

def main():
    parser = argparse.ArgumentParser(description="Setup Crime Hotspot Prediction System")
    parser.add_argument("--docker", action="store_true", help="Setup with Docker")
    parser.add_argument("--local", action="store_true", help="Setup for local development")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data")
    
    args = parser.parse_args()
    
    print("🚔 Crime Hotspot Prediction System Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Create sample data if requested
    if args.sample_data:
        download_sample_data()
    
    # Setup based on preference
    if args.docker:
        print("\n🐳 Setting up Docker environment...")
        setup_docker()
        print("\nTo run with Docker:")
        print("docker-compose up --build")
        
    elif args.local:
        print("\n🐍 Setting up local development environment...")
        setup_virtual_environment()
        print("\nTo activate virtual environment:")
        if os.name == 'nt':
            print("venv\\Scripts\\activate")
        else:
            print("source venv/bin/activate")
        print("\nTo run locally:")
        print("python app.py")
        
    else:
        print("\nChoose setup method:")
        print("--docker: Setup with Docker")
        print("--local: Setup for local development")
        return
    
    # Run tests if requested
    if args.test:
        print("\n🧪 Running tests...")
        run_tests()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Place your Crime_Incidents_in_2024.csv file in the data/ directory")
    print("2. Train the model: python app.py train data/Crime_Incidents_in_2024.csv")
    print("3. Start the API: python app.py (or docker-compose up)")
    print("4. Visit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()