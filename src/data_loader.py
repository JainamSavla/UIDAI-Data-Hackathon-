"""
Data Loading Module for UIDAI Hackathon Analytics Pipeline
Handles data ingestion, validation, and initial exploration
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from src.utils import DataValidation, get_memory_usage


class DataLoader:
    """
    Load and validate UIDAI Aadhaar enrollment data.
    
    Key Responsibilities:
    - Load CSV data from multiple sources
    - Validate data integrity
    - Initial data exploration
    - Handle data quality issues
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize DataLoader.
        
        Args:
            logger: Logging instance
        """
        self.logger = logger
        self.data = None
        self.original_shape = None
        self.column_info = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load Aadhaar enrollment data from CSV file.
        
        Args:
            data_path: Path to CSV file or directory containing CSV files
        
        Returns:
            Loaded dataframe
        """
        self.logger.info(f"Loading data from: {data_path}")
        
        try:
            if os.path.isdir(data_path):
                # Load multiple CSV files from directory
                csv_files = list(Path(data_path).glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found in {data_path}")
                
                self.logger.info(f"Found {len(csv_files)} CSV files")
                dfs = []
                for csv_file in sorted(csv_files):
                    self.logger.info(f"Loading {csv_file.name}...")
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    self.logger.info(f"  Loaded {len(df)} rows")
                
                self.data = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Total rows after concatenation: {len(self.data)}")
            else:
                # Load single CSV file
                self.data = pd.read_csv(data_path)
                self.logger.info(f"Loaded {len(self.data)} rows")
            
            self.original_shape = self.data.shape
            return self.data
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate loaded data integrity and quality.
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Starting data validation...")
        validation_results = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicates": self.data.duplicated().sum(),
            "memory_usage": get_memory_usage(self.data)
        }
        
        # Log validation results
        self.logger.info(f"Shape: {validation_results['shape']}")
        self.logger.info(f"Columns: {len(validation_results['columns'])}")
        self.logger.info(f"Duplicates: {validation_results['duplicates']}")
        self.logger.info(f"Memory usage: {validation_results['memory_usage']}")
        
        return validation_results
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform initial exploratory analysis.
        
        Returns:
            Dictionary with exploration results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Exploring data...")
        exploration = {
            "head": self.data.head(5).to_dict('records'),
            "tail": self.data.tail(5).to_dict('records'),
            "data_types": self.data.dtypes.to_dict(),
            "null_counts": self.data.isnull().sum().to_dict(),
            "numeric_summary": {}
        }
        
        # Numeric columns summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            exploration["numeric_summary"][col] = {
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max()),
                "mean": float(self.data[col].mean()),
                "median": float(self.data[col].median()),
                "std": float(self.data[col].std())
            }
        
        self.logger.info(f"Numeric columns: {list(numeric_cols)}")
        return exploration
    
    def get_column_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about each column.
        
        Returns:
            Dictionary with column information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.column_info = {}
        for col in self.data.columns:
            dtype = self.data[col].dtype
            non_null = self.data[col].notna().sum()
            null_count = self.data[col].isna().sum()
            
            info = {
                "dtype": str(dtype),
                "non_null_count": int(non_null),
                "null_count": int(null_count),
                "null_percentage": float(null_count / len(self.data) * 100),
                "unique_count": int(self.data[col].nunique())
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.data[col]):
                info["min"] = float(self.data[col].min())
                info["max"] = float(self.data[col].max())
                info["mean"] = float(self.data[col].mean())
                info["median"] = float(self.data[col].median())
            
            # Add top values for categorical columns
            if pd.api.types.is_object_dtype(self.data[col]):
                info["top_values"] = self.data[col].value_counts().head(3).to_dict()
            
            self.column_info[col] = info
        
        return self.column_info
    
    def identify_date_columns(self) -> list:
        """
        Identify potential date columns.
        
        Returns:
            List of date column names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        date_cols = []
        date_keywords = ['date', 'time', 'enrol', 'update', 'created', 'modified']
        
        for col in self.data.columns:
            # Check column name
            if any(keyword in col.lower() for keyword in date_keywords):
                date_cols.append(col)
            # Try to parse as date
            elif pd.api.types.is_object_dtype(self.data[col]):
                try:
                    pd.to_datetime(self.data[col].dropna().head(100))
                    date_cols.append(col)
                except:
                    pass
        
        self.logger.info(f"Identified date columns: {date_cols}")
        return date_cols
    
    def identify_categorical_columns(self) -> list:
        """
        Identify categorical columns.
        
        Returns:
            List of categorical column names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        categorical_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object' or self.data[col].nunique() < 50:
                categorical_cols.append(col)
        
        self.logger.info(f"Identified categorical columns: {categorical_cols}")
        return categorical_cols
    
    def get_data(self) -> pd.DataFrame:
        """
        Get loaded data.
        
        Returns:
            Loaded dataframe
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "memory_mb": get_memory_usage(self.data),
            "duplicates": int(self.data.duplicated().sum()),
            "missing_values_total": int(self.data.isnull().sum().sum()),
            "date_columns": self.identify_date_columns(),
            "categorical_columns": self.identify_categorical_columns(),
            "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns)
        }
        
        return summary


def load_enrollment_data(data_path: str, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load and validate enrollment data.
    
    Args:
        data_path: Path to data file or directory
        logger: Logging instance
    
    Returns:
        Tuple of (dataframe, validation_results)
    """
    loader = DataLoader(logger)
    data = loader.load_data(data_path)
    validation = loader.validate_data()
    
    return data, validation
