"""
Utility functions for UIDAI Hackathon Analytics Pipeline
Includes logging configuration and helper functions
"""

import logging
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd


def setup_logging(log_dir: str = None) -> logging.Logger:
    """
    Configure logging for the entire pipeline.
    
    Args:
        log_dir: Directory to save logs. If None, uses current directory.
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = os.getcwd()
    
    log_path = os.path.join(log_dir, "uidai_analysis.log")
    
    # Create logger
    logger = logging.getLogger("uidai_hackathon")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Logs saved to: {log_path}")
    return logger


def ensure_output_dir(output_dir: str = "outputs") -> str:
    """
    Ensure output directory exists.
    
    Args:
        output_dir: Path to output directory
    
    Returns:
        Absolute path to output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return os.path.abspath(output_dir)


def save_dict_to_json(data: dict, filepath: str, logger: logging.Logger = None) -> None:
    """
    Save dictionary to JSON file with proper formatting.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        logger: Optional logger instance
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        if logger:
            logger.info(f"Saved analysis results to {filepath}")
    except Exception as e:
        if logger:
            logger.error(f"Error saving JSON: {str(e)}")
        raise


def print_section_header(title: str, logger: logging.Logger = None) -> None:
    """
    Print formatted section header for console output.
    
    Args:
        title: Section title
        logger: Optional logger instance
    """
    header = f"\n{'='*80}\n  {title}\n{'='*80}\n"
    print(header)
    if logger:
        logger.info(title)


def print_insights(insights: dict, logger: logging.Logger = None) -> None:
    """
    Pretty print insights dictionary.
    
    Args:
        insights: Dictionary of insights
        logger: Optional logger instance
    """
    for key, value in insights.items():
        line = f"  • {key}: {value}"
        print(line)
        if logger:
            logger.info(line)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling zero division.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def calculate_percentile_stats(series: pd.Series, percentiles: list = None) -> dict:
    """
    Calculate percentile statistics for a series.
    
    Args:
        series: Pandas series
        percentiles: List of percentiles (0-100)
    
    Returns:
        Dictionary of percentile values
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    
    stats = {}
    for p in percentiles:
        stats[f'p{p}'] = series.quantile(p / 100)
    
    return stats


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop", 
                          logger: logging.Logger = None) -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df: Input dataframe
        strategy: 'drop' or 'forward_fill'
        logger: Optional logger instance
    
    Returns:
        Cleaned dataframe
    """
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        if logger:
            logger.info("No missing values found")
        return df
    
    if logger:
        logger.warning(f"Found {missing_count} missing values")
    
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "forward_fill":
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    if logger:
        logger.info(f"Applied '{strategy}' strategy. Remaining rows: {len(df)}")
    
    return df


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of dataframe.
    
    Args:
        df: Pandas dataframe
    
    Returns:
        Formatted memory usage string
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    return f"{memory_mb:.2f} MB"


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with thousand separators.
    
    Args:
        num: Number to format
        decimals: Decimal places
    
    Returns:
        Formatted number string
    """
    return f"{num:,.{decimals}f}"


def get_timestamp() -> str:
    """
    Get current timestamp in standard format.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DataValidation:
    """Helper class for data validation"""
    
    @staticmethod
    def validate_date_column(df: pd.DataFrame, col: str, logger: logging.Logger = None) -> bool:
        """
        Validate if column contains valid dates.
        
        Args:
            df: Dataframe
            col: Column name
            logger: Optional logger
        
        Returns:
            True if valid, False otherwise
        """
        try:
            pd.to_datetime(df[col])
            if logger:
                logger.info(f"Column '{col}' validated as datetime")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Column '{col}' validation failed: {str(e)}")
            return False
    
    @staticmethod
    def validate_numeric_column(df: pd.DataFrame, col: str, logger: logging.Logger = None) -> bool:
        """
        Validate if column contains numeric values.
        
        Args:
            df: Dataframe
            col: Column name
            logger: Optional logger
        
        Returns:
            True if valid, False otherwise
        """
        try:
            pd.to_numeric(df[col])
            if logger:
                logger.info(f"Column '{col}' validated as numeric")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Column '{col}' validation failed: {str(e)}")
            return False


class AnalysisResults:
    """Container for analysis results with easy export"""
    
    def __init__(self):
        """Initialize results container"""
        self.descriptive = {}
        self.diagnostic = {}
        self.predictive = {}
        self.prescriptive = {}
        self.metadata = {
            "created_at": get_timestamp(),
            "pipeline_version": "1.0"
        }
    
    def to_dict(self) -> dict:
        """Convert results to dictionary"""
        return {
            "metadata": self.metadata,
            "descriptive_analytics": self.descriptive,
            "diagnostic_analytics": self.diagnostic,
            "predictive_analytics": self.predictive,
            "prescriptive_analytics": self.prescriptive
        }
    
    def save_to_json(self, filepath: str, logger: logging.Logger = None) -> None:
        """Save all results to JSON file"""
        save_dict_to_json(self.to_dict(), filepath, logger)


# Global logger instance
logger = None


def get_logger(log_dir: str = None) -> logging.Logger:
    """
    Get or initialize global logger.
    
    Args:
        log_dir: Directory for logs
    
    Returns:
        Logger instance
    """
    global logger
    if logger is None:
        logger = setup_logging(log_dir)
    return logger
