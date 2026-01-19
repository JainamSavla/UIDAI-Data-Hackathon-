"""
Descriptive Analytics Module for UIDAI Hackathon Analytics Pipeline
Provides comprehensive statistical summaries and pattern identification
Addresses UIDAI Judging Criteria: Data Analysis & Insights
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy import stats


class DescriptiveAnalytics:
    """
    Perform descriptive analytics on UIDAI Aadhaar enrollment data.
    
    Key Responsibilities:
    - Age distribution analysis
    - Temporal trend analysis
    - Geographic pattern analysis
    - Demographic composition
    - Enrollment patterns by category
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize DescriptiveAnalytics.
        
        Args:
            logger: Logging instance
        """
        self.logger = logger
        self.data = None
        self.results = {}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete descriptive analytics.
        
        Args:
            df: Input dataframe with engineered features
        
        Returns:
            Dictionary with all analysis results
        """
        self.data = df
        self.logger.info("Starting descriptive analytics...")
        
        # Overall statistics
        self.results['overall_statistics'] = self._analyze_overall_statistics()
        
        # Age distribution analysis
        if 'age' in self.data.columns:
            self.results['age_distribution'] = self._analyze_age_distribution()
        
        # Gender distribution
        if 'gender' in self.data.columns:
            self.results['gender_distribution'] = self._analyze_gender_distribution()
        
        # Temporal trends
        date_cols = [col for col in self.data.columns if 'date' in col.lower()]
        if date_cols and 'enrollment_month' in self.data.columns:
            self.results['temporal_trends'] = self._analyze_temporal_trends()
        
        # Geographic analysis
        if 'state' in [col.lower() for col in self.data.columns]:
            self.results['geographic_analysis'] = self._analyze_geographic_patterns()
        
        # Enrollment patterns
        if 'age_group' in self.data.columns:
            self.results['enrollment_patterns'] = self._analyze_enrollment_patterns()
        
        self.logger.info("Descriptive analytics complete")
        return self.results
    
    def _analyze_overall_statistics(self) -> Dict[str, Any]:
        """Analyze overall enrollment statistics."""
        stats_dict = {
            "total_enrollments": len(self.data),
            "unique_age_values": int(self.data['age'].nunique()) if 'age' in self.data.columns else 0,
            "age_statistics": {}
        }
        
        if 'age' in self.data.columns:
            stats_dict["age_statistics"] = {
                "min_age": int(self.data['age'].min()),
                "max_age": int(self.data['age'].max()),
                "mean_age": round(float(self.data['age'].mean()), 2),
                "median_age": float(self.data['age'].median()),
                "std_age": round(float(self.data['age'].std()), 2),
                "q25_age": float(self.data['age'].quantile(0.25)),
                "q75_age": float(self.data['age'].quantile(0.75))
            }
        
        return stats_dict
    
    def _analyze_age_distribution(self) -> Dict[str, Any]:
        """
        Analyze age distribution with focus on vulnerable populations.
        
        HACKATHON CRITERIA: Identifies meaningful patterns (vulnerable groups)
        """
        analysis = {
            "age_group_distribution": {},
            "vulnerable_group_analysis": {},
            "insights": []
        }
        
        # Age group distribution
        if 'age_group' in self.data.columns:
            age_group_counts = self.data['age_group'].value_counts().sort_index()
            age_group_pct = self.data['age_group'].value_counts(normalize=True).sort_index() * 100
            
            for group in age_group_counts.index:
                analysis["age_group_distribution"][str(group)] = {
                    "count": int(age_group_counts[group]),
                    "percentage": round(float(age_group_pct[group]), 2)
                }
        
        # Vulnerable group analysis (0-5 and 60+)
        if 'is_vulnerable_group' in self.data.columns:
            vulnerable_count = self.data['is_vulnerable_group'].sum()
            vulnerable_pct = (vulnerable_count / len(self.data)) * 100
            
            analysis["vulnerable_group_analysis"] = {
                "vulnerable_count": int(vulnerable_count),
                "vulnerable_percentage": round(float(vulnerable_pct), 2),
                "children_0_5": int(len(self.data[self.data['age_group'] == '0-5 Years'])) if 'age_group' in self.data.columns else 0,
                "elderly_60_plus": int(len(self.data[self.data['age_group'] == '60+ Years'])) if 'age_group' in self.data.columns else 0
            }
            
            analysis["insights"].append(f"Vulnerable populations (0-5 and 60+) represent {vulnerable_pct:.1f}% of enrollments")
        
        # Age skewness
        age_skew = float(self.data['age'].skew())
        analysis["age_distribution_skewness"] = round(age_skew, 3)
        if age_skew > 0.5:
            analysis["insights"].append("Age distribution is right-skewed (younger population)")
        elif age_skew < -0.5:
            analysis["insights"].append("Age distribution is left-skewed (older population)")
        
        return analysis
    
    def _analyze_gender_distribution(self) -> Dict[str, Any]:
        """Analyze gender-based distribution."""
        analysis = {
            "gender_distribution": {},
            "insights": []
        }
        
        if 'gender' in self.data.columns:
            gender_counts = self.data['gender'].value_counts()
            gender_pct = self.data['gender'].value_counts(normalize=True) * 100
            
            for gender in gender_counts.index:
                analysis["gender_distribution"][str(gender)] = {
                    "count": int(gender_counts[gender]),
                    "percentage": round(float(gender_pct[gender]), 2)
                }
            
            # Gender-age cross-tabulation
            if 'age_group' in self.data.columns:
                cross_tab = pd.crosstab(
                    self.data['gender'],
                    self.data['age_group'],
                    normalize='index'
                ) * 100
                analysis["gender_age_distribution"] = cross_tab.round(2).to_dict()
        
        return analysis
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """
        Analyze temporal enrollment trends.
        
        HACKATHON CRITERIA: Identifies trends over time
        """
        analysis = {
            "monthly_trends": {},
            "yearly_trends": {},
            "insights": []
        }
        
        # Monthly enrollment trends
        if 'enrollment_month' in self.data.columns:
            monthly_counts = self.data['enrollment_month'].value_counts().sort_index()
            for month in monthly_counts.index:
                month_name = pd.Timestamp(year=2000, month=int(month), day=1).strftime('%B')
                analysis["monthly_trends"][month_name] = int(monthly_counts[month])
        
        # Yearly enrollment trends
        if 'enrollment_year' in self.data.columns:
            yearly_counts = self.data['enrollment_year'].value_counts().sort_index()
            for year in yearly_counts.index:
                analysis["yearly_trends"][int(year)] = int(yearly_counts[year])
        
        # Day of week analysis
        if 'enrollment_day_name' in self.data.columns:
            dow_counts = self.data['enrollment_day_name'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = dow_counts.reindex([d for d in day_order if d in dow_counts.index])
            
            weekend_count = self.data['is_weekend'].sum() if 'is_weekend' in self.data.columns else 0
            weekday_count = len(self.data) - weekend_count
            analysis["weekend_vs_weekday"] = {
                "weekday_enrollments": int(weekday_count),
                "weekend_enrollments": int(weekend_count),
                "weekend_percentage": round((weekend_count / len(self.data)) * 100, 2)
            }
        
        return analysis
    
    def _analyze_geographic_patterns(self) -> Dict[str, Any]:
        """
        Analyze geographic enrollment patterns.
        
        HACKATHON CRITERIA: Identifies geographic disparities
        """
        analysis = {
            "state_distribution": {},
            "district_distribution": {},
            "geographic_concentration": {},
            "insights": []
        }
        
        # Find state column
        state_col = None
        for col in self.data.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if state_col:
            state_counts = self.data[state_col].value_counts()
            for state in state_counts.head(10).index:
                analysis["state_distribution"][str(state)] = {
                    "count": int(state_counts[state]),
                    "percentage": round((state_counts[state] / len(self.data)) * 100, 2)
                }
            
            # Herfindahl index for concentration
            state_shares = state_counts / len(self.data)
            hhi = (state_shares ** 2).sum()
            analysis["geographic_concentration"]["herfindahl_index"] = round(float(hhi), 4)
            
            if hhi > 0.25:
                analysis["insights"].append("High geographic concentration - few states dominate")
            else:
                analysis["insights"].append("Enrollment distributed across multiple states")
        
        # District distribution
        district_col = None
        for col in self.data.columns:
            if 'district' in col.lower():
                district_col = col
                break
        
        if district_col:
            district_counts = self.data[district_col].value_counts()
            analysis["district_distribution"] = {
                "total_districts": int(self.data[district_col].nunique()),
                "top_10_districts": district_counts.head(10).to_dict()
            }
        
        return analysis
    
    def _analyze_enrollment_patterns(self) -> Dict[str, Any]:
        """Analyze enrollment patterns by demographics."""
        analysis = {
            "age_group_trends": {},
            "enrollment_velocity": {},
            "insights": []
        }
        
        # Age group temporal patterns
        if 'age_group' in self.data.columns and 'enrollment_year' in self.data.columns:
            age_year_crosstab = pd.crosstab(
                self.data['enrollment_year'],
                self.data['age_group']
            )
            analysis["age_group_trends"] = age_year_crosstab.to_dict()
        
        # Recent vs older enrollments
        if 'is_recent_enrollment' in self.data.columns:
            recent_count = self.data['is_recent_enrollment'].sum()
            older_count = len(self.data) - recent_count
            analysis["enrollment_velocity"] = {
                "recent_6_months": int(recent_count),
                "older_than_6_months": int(older_count),
                "recent_percentage": round((recent_count / len(self.data)) * 100, 2)
            }
        
        return analysis
    
    def get_results(self) -> Dict[str, Any]:
        """Get all descriptive analysis results."""
        return self.results
    
    def print_summary(self) -> None:
        """Print formatted summary of findings."""
        self.logger.info("\n=== DESCRIPTIVE ANALYTICS SUMMARY ===\n")
        
        # Overall stats
        overall = self.results.get('overall_statistics', {})
        self.logger.info(f"Total Enrollments: {overall.get('total_enrollments', 'N/A')}")
        age_stats = overall.get('age_statistics', {})
        self.logger.info(f"Mean Age: {age_stats.get('mean_age', 'N/A')}")
        
        # Age distribution
        age_dist = self.results.get('age_distribution', {})
        if 'insights' in age_dist:
            for insight in age_dist['insights']:
                self.logger.info(f"  • {insight}")


def perform_descriptive_analysis(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Convenience function to perform descriptive analytics.
    
    Args:
        df: Input dataframe with engineered features
        logger: Logging instance
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = DescriptiveAnalytics(logger)
    results = analyzer.analyze(df)
    analyzer.print_summary()
    
    return results
