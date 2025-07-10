#!/usr/bin/env python3
"""
Data cleaning script for Crime Incidents data
Fixes common data quality issues before training
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_crime_data(input_path, output_path=None):
    """Clean and prepare crime incident data"""
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Original data shape: {df.shape}")
    
    # 1. Handle date columns
    logger.info("Cleaning date columns...")
    df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], errors='coerce')
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')
    
    # 2. Clean geographical data
    logger.info("Cleaning geographical data...")
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # Filter out obviously wrong coordinates
    df = df[(df['LATITUDE'] > 35) & (df['LATITUDE'] < 45)]  # Reasonable range for DC area
    df = df[(df['LONGITUDE'] > -80) & (df['LONGITUDE'] < -70)]  # Reasonable range for DC area
    
    # 3. Clean categorical columns
    logger.info("Cleaning categorical columns...")
    
    # SHIFT: Make consistent
    df['SHIFT'] = df['SHIFT'].fillna('UNKNOWN')
    df['SHIFT'] = df['SHIFT'].str.upper()
    
    # METHOD: Clean up
    df['METHOD'] = df['METHOD'].fillna('UNKNOWN')
    df['METHOD'] = df['METHOD'].str.upper()
    
    # OFFENSE: Clean up
    df['OFFENSE'] = df['OFFENSE'].fillna('UNKNOWN')
    df['OFFENSE'] = df['OFFENSE'].str.upper()
    
    # ANC: Handle mixed types
    df['ANC'] = df['ANC'].astype(str).fillna('UNKNOWN')
    df['ANC'] = df['ANC'].replace('nan', 'UNKNOWN')
    
    # NEIGHBORHOOD_CLUSTER: Clean up
    df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(str).fillna('UNKNOWN')
    df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].replace('nan', 'UNKNOWN')
    
    # 4. Clean numerical columns
    logger.info("Cleaning numerical columns...")
    
    # WARD: Convert to integer
    df['WARD'] = pd.to_numeric(df['WARD'], errors='coerce').fillna(0).astype(int)
    
    # DISTRICT: Convert to integer
    df['DISTRICT'] = pd.to_numeric(df['DISTRICT'], errors='coerce').fillna(0).astype(int)
    
    # PSA: Convert to integer
    df['PSA'] = pd.to_numeric(df['PSA'], errors='coerce').fillna(0).astype(int)
    
    # 5. Handle other problematic columns
    logger.info("Cleaning other columns...")
    
    # BID: Handle empty values
    df['BID'] = df['BID'].fillna('UNKNOWN')
    
    # VOTING_PRECINCT: Handle mixed types
    df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(str).fillna('UNKNOWN')
    df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].replace('nan', 'UNKNOWN')
    
    # BLOCK_GROUP: Handle mixed types
    df['BLOCK_GROUP'] = df['BLOCK_GROUP'].astype(str).fillna('UNKNOWN')
    df['BLOCK_GROUP'] = df['BLOCK_GROUP'].replace('nan', 'UNKNOWN')
    
    # 6. Remove rows with critical missing data
    logger.info("Removing rows with critical missing data...")
    initial_rows = len(df)
    df = df.dropna(subset=['REPORT_DAT', 'OFFENSE', 'LATITUDE', 'LONGITUDE'])
    final_rows = len(df)
    
    logger.info(f"Removed {initial_rows - final_rows} rows with missing critical data")
    logger.info(f"Final data shape: {df.shape}")
    
    # 7. Save cleaned data
    if output_path is None:
        output_path = input_path.replace('.csv', '_cleaned.csv')
    
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    
    # 8. Generate summary report
    logger.info("Data cleaning summary:")
    logger.info(f"  Original rows: {initial_rows}")
    logger.info(f"  Final rows: {final_rows}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['REPORT_DAT'].min()} to {df['REPORT_DAT'].max()}")
    logger.info(f"  Unique offenses: {df['OFFENSE'].nunique()}")
    logger.info(f"  Unique districts: {df['DISTRICT'].nunique()}")
    logger.info(f"  Unique wards: {df['WARD'].nunique()}")
    
    return df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean crime incident data")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Clean the data
    try:
        clean_crime_data(args.input, args.output)
        logger.info("✅ Data cleaning completed successfully!")
        
        if args.output:
            logger.info(f"You can now use the cleaned data: {args.output}")
        else:
            cleaned_file = args.input.replace('.csv', '_cleaned.csv')
            logger.info(f"You can now use the cleaned data: {cleaned_file}")
            
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()