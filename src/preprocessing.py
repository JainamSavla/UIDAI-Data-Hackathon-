"""
Data Preprocessing Module for UIDAI Hackathon Analytics Pipeline
Handles data cleaning, transformation, and standardization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any
from datetime import datetime


class DataPreprocessor:
    """
    Preprocess UIDAI Aadhaar enrollment data.
    
    Key Responsibilities:
    - Data cleaning and validation
    - Handle missing values
    - Standardize data formats
    - Remove outliers and duplicates
    - Data transformation for analysis
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize DataPreprocessor.
        
        Args:
            logger: Logging instance
        """
        self.logger = logger
        self.data = None
        self.original_data = None
        self.preprocessing_steps = []
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
        
        Returns:
            Preprocessed dataframe
        """
        self.data = df.copy()
        self.original_data = df.copy()
        self.logger.info(f"Starting preprocessing on {len(self.data)} rows")
        
        # Step 1: Remove duplicates
        self.data = self._remove_duplicates()
        
        # Step 2: Handle missing values
        self.data = self._handle_missing_values()
        
        # Step 3: Parse and standardize dates
        self.data = self._parse_dates()
        
        # Step 4: Standardize column names
        self.data = self._standardize_columns()
        
        # Step 5: Remove invalid records
        self.data = self._remove_invalid_records()
        
        # Step 6: Data type optimization
        self.data = self._optimize_dtypes()
        
        self.logger.info(f"Preprocessing complete. Final shape: {self.data.shape}")
        self.logger.info(f"Preprocessing steps: {self.preprocessing_steps}")
        
        return self.data
    
    def _remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(self.data)
        data = self.data.drop_duplicates()
        removed = initial_rows - len(data)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate rows")
            self.preprocessing_steps.append(f"Removed {removed} duplicates")
        
        return data
    
    def _handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        data = self.data.copy()
        missing_summary = data.isnull().sum()
        
        if missing_summary.sum() > 0:
            self.logger.info(f"Found missing values: {missing_summary[missing_summary > 0].to_dict()}")
            
            # Drop rows with missing values in critical columns
            critical_cols = ['date', 'state', 'district', 'age', 'gender']
            critical_cols = [col for col in critical_cols if col in data.columns]
            
            if critical_cols:
                data = data.dropna(subset=critical_cols)
                self.logger.info(f"Dropped rows with missing critical values. New shape: {data.shape}")
                self.preprocessing_steps.append(f"Dropped rows with missing critical values")
            
            # Forward fill for temporal data if date column exists
            date_cols = [col for col in data.columns if 'date' in col.lower() and col not in critical_cols]
            for col in date_cols:
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _parse_dates(self) -> pd.DataFrame:
        """Parse and standardize date columns."""
        data = self.data.copy()
        
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        for col in date_cols:
            try:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                self.logger.info(f"Parsed date column: {col}")
                self.preprocessing_steps.append(f"Parsed date column: {col}")
            except Exception as e:
                self.logger.warning(f"Could not parse {col}: {str(e)}")
        
        return data
    
    def _standardize_columns(self) -> pd.DataFrame:
        """Standardize column names and values."""
        data = self.data.copy()
        
        # Lowercase and clean column names
        data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Standardize text columns
        string_cols = data.select_dtypes(include=['object']).columns
        for col in string_cols:
            data[col] = data[col].str.strip().str.title()
        
        self.logger.info(f"Standardized column names: {list(data.columns)}")
        self.preprocessing_steps.append("Standardized column names and text values")
        
        return data
    
    def _remove_invalid_records(self) -> pd.DataFrame:
        """Remove invalid records based on logical constraints."""
        data = self.data.copy()
        initial_rows = len(data)
        
        # Remove records with invalid age
        if 'age' in data.columns:
            data = data[(data['age'] >= 0) & (data['age'] <= 120)]
        
        # Remove future dates
        date_cols = data.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            data = data[data[col] <= pd.Timestamp.now()]
        
        removed = initial_rows - len(data)
        if removed > 0:
            self.logger.info(f"Removed {removed} invalid records")
            self.preprocessing_steps.append(f"Removed {removed} invalid records")
        
        return data
    
    def _optimize_dtypes(self) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        data = self.data.copy()
        
        # Convert integer columns
        int_cols = data.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if data[col].min() >= 0:
                if data[col].max() < 256:
                    data[col] = data[col].astype('uint8')
                elif data[col].max() < 65536:
                    data[col] = data[col].astype('uint16')
                else:
                    data[col] = data[col].astype('uint32')
            else:
                data[col] = data[col].astype('int32')
        
        # Convert float columns
        float_cols = data.select_dtypes(include=['float64']).columns
        for col in float_cols:
            data[col] = data[col].astype('float32')
        
        # Convert low-cardinality string columns to category
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) < 0.05:
                data[col] = data[col].astype('category')
        
        self.logger.info("Optimized data types")
        self.preprocessing_steps.append("Optimized data types")
        
        return data
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """
        Generate preprocessing report.
        
        Returns:
            Dictionary with preprocessing statistics
        """
        report = {
            "original_shape": self.original_data.shape,
            "final_shape": self.data.shape,
            "rows_removed": self.original_data.shape[0] - self.data.shape[0],
            "rows_removed_percentage": (
                (self.original_data.shape[0] - self.data.shape[0]) / 
                self.original_data.shape[0] * 100
            ),
            "preprocessing_steps": self.preprocessing_steps,
            "missing_values_before": int(self.original_data.isnull().sum().sum()),
            "missing_values_after": int(self.data.isnull().sum().sum())
        }
        
        return report
    
    def get_data(self) -> pd.DataFrame:
        """Get preprocessed data."""
        if self.data is None:
            raise ValueError("No data preprocessed. Call preprocess() first.")
        return self.data


def preprocess_enrollment_data(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to preprocess enrollment data.
    
    Args:
        df: Input dataframe
        logger: Logging instance
    
    Returns:
        Tuple of (preprocessed_dataframe, preprocessing_report)
    """
    preprocessor = DataPreprocessor(logger)
    data = preprocessor.preprocess(df)
    report = preprocessor.get_preprocessing_report()
    
    return data, report
