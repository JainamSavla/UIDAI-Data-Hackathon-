"""
Feature Engineering Module for UIDAI Hackathon Analytics Pipeline
Creates meaningful features from raw data for analysis and modeling
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Engineer features from UIDAI Aadhaar enrollment data.
    
    Key Responsibilities:
    - Create age group categories
    - Extract temporal features
    - Create geographic features
    - Identify enrollment patterns
    - Create interaction features
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize FeatureEngineer.
        
        Args:
            logger: Logging instance
        """
        self.logger = logger
        self.data = None
        self.original_data = None
        self.engineered_features = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        
        Args:
            df: Input dataframe (after preprocessing)
        
        Returns:
            Dataframe with engineered features
        """
        self.data = df.copy()
        self.original_data = df.copy()
        self.logger.info("Starting feature engineering...")
        
        # Age-based features - handle both single age column and age group counts
        if 'age' in self.data.columns:
            self.data = self._create_age_features()
        elif 'demo_age_5_17' in self.data.columns and 'demo_age_17_' in self.data.columns:
            # Create age_group based on which count is higher
            self.data['age_group'] = 'Unknown'
            # Mark records as predominantly one age group or another based on counts
            # This is a simplified approach since we don't have individual ages
            self.data.loc[self.data['demo_age_5_17'] > self.data['demo_age_17_'], 'age_group'] = '5-17 Years'
            self.data.loc[self.data['demo_age_17_'] > self.data['demo_age_5_17'], 'age_group'] = '17+ Years'
            self.data.loc[self.data['demo_age_5_17'] == self.data['demo_age_17_'], 'age_group'] = 'Mixed'
            self.engineered_features.append('age_group')
            self.logger.info("Created age_group feature from demographic counts")
        
        # Temporal features
        date_cols = [col for col in self.data.columns if 'date' in col.lower()]
        if date_cols:
            self.data = self._create_temporal_features(date_cols[0])
        
        # Geographic features
        geo_cols = [col for col in self.data.columns if any(
            keyword in col.lower() for keyword in ['state', 'district', 'pincode']
        )]
        if geo_cols:
            self.data = self._create_geographic_features(geo_cols)
        
        # Interaction features
        self.data = self._create_interaction_features()
        
        self.logger.info(f"Feature engineering complete. Total new features: {len(self.engineered_features)}")
        self.logger.info(f"New features: {self.engineered_features}")
        
        return self.data
    
    def _create_age_features(self) -> pd.DataFrame:
        """Create age-based categorical features."""
        data = self.data.copy()
        
        # Age groups: 0-5, 5-17, 18-35, 35-60, 60+
        data['age_group'] = pd.cut(
            data['age'],
            bins=[0, 5, 17, 35, 60, 150],
            labels=['0-5 Years', '5-17 Years', '18-35 Years', '35-60 Years', '60+ Years'],
            right=False
        )
        self.engineered_features.append('age_group')
        self.logger.info("Created age_group feature")
        
        # Age categories for detailed analysis
        data['age_category'] = pd.cut(
            data['age'],
            bins=[0, 1, 5, 12, 18, 30, 45, 60, 150],
            labels=['0-1', '1-5', '5-12', '12-18', '18-30', '30-45', '45-60', '60+'],
            right=False
        )
        self.engineered_features.append('age_category')
        
        # Is vulnerable group (children 0-5 and elderly 60+)
        data['is_vulnerable_group'] = data['age_group'].isin(['0-5 Years', '60+ Years']).astype(int)
        self.engineered_features.append('is_vulnerable_group')
        
        # Is working age (18-60)
        data['is_working_age'] = ((data['age'] >= 18) & (data['age'] < 60)).astype(int)
        self.engineered_features.append('is_working_age')
        
        return data
    
    def _create_temporal_features(self, date_col: str) -> pd.DataFrame:
        """Create temporal features from date column."""
        data = self.data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Extract temporal components
        data['enrollment_year'] = data[date_col].dt.year
        self.engineered_features.append('enrollment_year')
        
        data['enrollment_month'] = data[date_col].dt.month
        self.engineered_features.append('enrollment_month')
        
        data['enrollment_quarter'] = data[date_col].dt.quarter
        self.engineered_features.append('enrollment_quarter')
        
        data['enrollment_day_of_week'] = data[date_col].dt.dayofweek
        self.engineered_features.append('enrollment_day_of_week')
        
        data['enrollment_day_name'] = data[date_col].dt.day_name()
        self.engineered_features.append('enrollment_day_name')
        
        data['enrollment_month_name'] = data[date_col].dt.month_name()
        self.engineered_features.append('enrollment_month_name')
        
        # Is weekend
        data['is_weekend'] = data[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        self.engineered_features.append('is_weekend')
        
        # Days since enrollment (reference: max date in data)
        max_date = data[date_col].max()
        data['days_since_enrollment'] = (max_date - data[date_col]).dt.days
        self.engineered_features.append('days_since_enrollment')
        
        # Is recent enrollment (last 6 months)
        six_months_ago = max_date - timedelta(days=180)
        data['is_recent_enrollment'] = (data[date_col] >= six_months_ago).astype(int)
        self.engineered_features.append('is_recent_enrollment')
        
        self.logger.info(f"Created temporal features from {date_col}")
        
        return data
    
    def _create_geographic_features(self, geo_cols: list) -> pd.DataFrame:
        """Create geographic features."""
        data = self.data.copy()
        
        # Standardize geographic column names
        for col in geo_cols:
            if col not in data.columns:
                continue
            
            if 'state' in col.lower():
                state_col = col
            elif 'district' in col.lower():
                district_col = col
            elif 'pincode' in col.lower():
                pincode_col = col
        
        # Create district-level statistics (proxy for area)
        if 'district' in [col.lower() for col in geo_cols]:
            district_col = [col for col in geo_cols if 'district' in col.lower()][0]
            district_enrollment_counts = data[district_col].value_counts()
            data['district_enrollment_volume'] = data[district_col].map(district_enrollment_counts)
            self.engineered_features.append('district_enrollment_volume')
            
            # Classify districts by enrollment volume
            quantiles = data['district_enrollment_volume'].quantile([0.33, 0.67])
            data['district_priority_tier'] = pd.cut(
                data['district_enrollment_volume'],
                bins=[0, quantiles[0.33], quantiles[0.67], float('inf')],
                labels=['Tier 3 (Low)', 'Tier 2 (Medium)', 'Tier 1 (High)']
            )
            self.engineered_features.append('district_priority_tier')
        
        # Create state-level statistics
        if 'state' in [col.lower() for col in geo_cols]:
            state_col = [col for col in geo_cols if 'state' in col.lower()][0]
            state_enrollment_counts = data[state_col].value_counts()
            data['state_enrollment_volume'] = data[state_col].map(state_enrollment_counts)
            self.engineered_features.append('state_enrollment_volume')
        
        # Urban proxy using pincode
        if 'pincode' in [col.lower() for col in geo_cols]:
            pincode_col = [col for col in geo_cols if 'pincode' in col.lower()][0]
            # Extract first digit of pincode (postal region)
            data['pincode_region'] = data[pincode_col].astype(str).str[0]
            self.engineered_features.append('pincode_region')
            
            # Count unique pincodes per district as urban density proxy
            if 'district' in [col.lower() for col in geo_cols]:
                district_col = [col for col in geo_cols if 'district' in col.lower()][0]
                pincode_diversity = data.groupby(district_col)[pincode_col].nunique()
                data['urban_density_proxy'] = data[district_col].map(pincode_diversity)
                self.engineered_features.append('urban_density_proxy')
        
        self.logger.info("Created geographic features")
        return data
    
    def _create_interaction_features(self) -> pd.DataFrame:
        """Create interaction and composite features."""
        data = self.data.copy()
        
        # Age-Time interactions
        if 'enrollment_year' in data.columns and 'age_group' in data.columns:
            # This creates a tuple representation that can be used for grouping
            self.logger.info("Created age-time interaction context")
        
        # Gender-Age interactions
        if 'gender' in data.columns and 'age_group' in data.columns:
            data['gender_age_group'] = data['gender'].astype(str) + '_' + data['age_group'].astype(str)
            self.engineered_features.append('gender_age_group')
        
        # Geographic-Enrollment recency
        if 'district_priority_tier' in data.columns and 'is_recent_enrollment' in data.columns:
            data['district_recent_enrollment'] = (
                data['district_priority_tier'].astype(str) + '_Recent'
            )
            self.engineered_features.append('district_recent_enrollment')
        
        return data
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of engineered features.
        
        Returns:
            Dictionary with feature information
        """
        summary = {
            "total_original_features": self.original_data.shape[1],
            "total_new_features": len(self.engineered_features),
            "total_features": self.data.shape[1],
            "engineered_features": self.engineered_features,
            "columns": list(self.data.columns)
        }
        
        return summary
    
    def get_data(self) -> pd.DataFrame:
        """Get data with engineered features."""
        if self.data is None:
            raise ValueError("No data engineered. Call engineer_features() first.")
        return self.data


def engineer_features(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to engineer features.
    
    Args:
        df: Input dataframe
        logger: Logging instance
    
    Returns:
        Tuple of (dataframe_with_features, feature_summary)
    """
    engineer = FeatureEngineer(logger)
    data = engineer.engineer_features(df)
    summary = engineer.get_feature_summary()
    
    return data, summary
