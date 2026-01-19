"""
Diagnostic Analytics Module for UIDAI Hackathon Analytics Pipeline
Investigates why patterns exist and identifies statistically significant relationships
Addresses UIDAI Judging Criteria: Meaningful patterns, anomalies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy import stats


class DiagnosticAnalytics:
    """
    Perform diagnostic analytics to understand drivers of enrollment patterns.
    
    Key Responsibilities:
    - Statistical hypothesis testing
    - Age-Geography interaction analysis
    - Time-Age lifecycle analysis
    - Urban-rural disparity investigation
    - Anomaly detection
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize DiagnosticAnalytics.
        
        Args:
            logger: Logging instance
        """
        self.logger = logger
        self.data = None
        self.results = {}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete diagnostic analytics.
        
        Args:
            df: Input dataframe with engineered features
        
        Returns:
            Dictionary with diagnostic analysis results
        """
        self.data = df
        self.logger.info("Starting diagnostic analytics...")
        
        # Statistical tests
        self.results['statistical_tests'] = self._perform_statistical_tests()
        
        # Age-Geography interactions
        if 'age_group' in self.data.columns:
            self.results['age_geography_interaction'] = self._analyze_age_geography_interaction()
        
        # Time-Age lifecycle
        if 'enrollment_year' in self.data.columns and 'age_group' in self.data.columns:
            self.results['time_age_lifecycle'] = self._analyze_time_age_lifecycle()
        
        # Urban-Rural disparity
        if 'district_priority_tier' in self.data.columns:
            self.results['geographic_disparities'] = self._analyze_geographic_disparities()
        
        # Anomaly detection
        self.results['anomalies'] = self._detect_anomalies()
        
        self.logger.info("Diagnostic analytics complete")
        return self.results
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical hypothesis tests.
        
        HACKATHON CRITERIA: Statistical rigor and evidence
        """
        tests_results = {
            "age_normality": {},
            "chi_square_tests": {},
            "anova_tests": {},
            "insights": []
        }
        
        # Test 1: Age distribution normality
        if 'age' in self.data.columns:
            statistic, p_value = stats.normaltest(self.data['age'].dropna())
            tests_results["age_normality"] = {
                "test": "D'Agostino-Pearson normality test",
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "is_normal": p_value > 0.05
            }
            if p_value < 0.05:
                tests_results["insights"].append(
                    "Age distribution significantly deviates from normal (supports non-parametric analysis)"
                )
        
        # Test 2: Chi-square test for independence (Age Group vs Gender)
        if 'age_group' in self.data.columns and 'gender' in self.data.columns:
            contingency_table = pd.crosstab(self.data['age_group'], self.data['gender'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            tests_results["chi_square_tests"]["age_gender_independence"] = {
                "chi_square_statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "degrees_of_freedom": int(dof),
                "significant": p_value < 0.05
            }
            
            if p_value < 0.05:
                tests_results["insights"].append(
                    "Age and gender are statistically dependent - enrollment patterns differ by gender"
                )
        
        # Test 3: ANOVA for enrollment volume across districts
        if 'district_enrollment_volume' in self.data.columns and 'age_group' in self.data.columns:
            groups = [group['district_enrollment_volume'].values 
                     for name, group in self.data.groupby('age_group')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            tests_results["anova_tests"]["age_group_enrollment_volume"] = {
                "f_statistic": round(float(f_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05
            }
            
            if p_value < 0.05:
                tests_results["insights"].append(
                    "District enrollment volume varies significantly across age groups (ANOVA p<0.05)"
                )
        
        return tests_results
    
    def _analyze_age_geography_interaction(self) -> Dict[str, Any]:
        """
        Analyze interaction between age and geography.
        
        HACKATHON CRITERIA: Multi-dimensional analysis
        """
        analysis = {
            "age_group_by_state": {},
            "age_group_by_district_tier": {},
            "insights": []
        }
        
        # Age group distribution by state
        state_col = None
        for col in self.data.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if state_col:
            crosstab = pd.crosstab(
                self.data[state_col],
                self.data['age_group'],
                normalize='index'
            ) * 100
            analysis["age_group_by_state"] = crosstab.round(2).to_dict()
            
            # Identify variation
            variation = crosstab.std()
            max_variation = variation.idxmax() if len(variation) > 0 else None
            if max_variation:
                analysis["insights"].append(
                    f"Age group '{max_variation}' shows highest geographic variation ({variation[max_variation]:.1f}%)"
                )
        
        # Age group distribution by district priority tier
        if 'district_priority_tier' in self.data.columns:
            crosstab_tier = pd.crosstab(
                self.data['district_priority_tier'],
                self.data['age_group'],
                normalize='index'
            ) * 100
            analysis["age_group_by_district_tier"] = crosstab_tier.round(2).to_dict()
        
        return analysis
    
    def _analyze_time_age_lifecycle(self) -> Dict[str, Any]:
        """
        Analyze how age cohorts evolve over time (lifecycle analysis).
        
        HACKATHON CRITERIA: Temporal patterns and lifecycle trends
        """
        analysis = {
            "age_group_year_trends": {},
            "recent_vs_older_age_distribution": {},
            "enrollment_growth_by_age": {},
            "insights": []
        }
        
        # Age group trends over years
        pivot = pd.crosstab(
            self.data['enrollment_year'],
            self.data['age_group']
        )
        analysis["age_group_year_trends"] = pivot.to_dict()
        
        # Recent vs older enrollments by age group
        if 'is_recent_enrollment' in self.data.columns:
            crosstab_recent = pd.crosstab(
                self.data['age_group'],
                self.data['is_recent_enrollment'],
                normalize='index'
            ) * 100
            analysis["recent_vs_older_age_distribution"] = crosstab_recent.round(2).to_dict()
            
            # Identify which age groups are enrolling recently
            recent_pct = crosstab_recent[1] if 1 in crosstab_recent.columns else {}
            if len(recent_pct) > 0:
                max_recent_age = recent_pct.idxmax()
                analysis["insights"].append(
                    f"Age group '{max_recent_age}' has highest recent enrollment rate ({recent_pct[max_recent_age]:.1f}%)"
                )
        
        # Growth trends by age group
        years = sorted(self.data['enrollment_year'].unique())
        if len(years) >= 2:
            for age_group in self.data['age_group'].unique():
                age_data = self.data[self.data['age_group'] == age_group]
                yearly_counts = age_data['enrollment_year'].value_counts().sort_index()
                
                if len(yearly_counts) >= 2:
                    first_year_count = yearly_counts.iloc[0]
                    last_year_count = yearly_counts.iloc[-1]
                    growth = ((last_year_count - first_year_count) / first_year_count * 100) if first_year_count > 0 else 0
                    
                    analysis["enrollment_growth_by_age"][str(age_group)] = {
                        "first_year_count": int(first_year_count),
                        "last_year_count": int(last_year_count),
                        "growth_percentage": round(float(growth), 2)
                    }
        
        return analysis
    
    def _analyze_geographic_disparities(self) -> Dict[str, Any]:
        """
        Analyze urban-rural and geographic disparities.
        
        HACKATHON CRITERIA: Identifies unenrolled populations and disparities
        """
        analysis = {
            "district_tier_analysis": {},
            "enrollment_concentration": {},
            "insights": []
        }
        
        # District tier analysis
        if 'district_priority_tier' in self.data.columns:
            tier_counts = self.data['district_priority_tier'].value_counts()
            tier_pct = self.data['district_priority_tier'].value_counts(normalize=True) * 100
            
            for tier in tier_counts.index:
                analysis["district_tier_analysis"][str(tier)] = {
                    "count": int(tier_counts[tier]),
                    "percentage": round(float(tier_pct[tier]), 2)
                }
            
            # Calculate disparity ratio (Tier 1 vs Tier 3)
            if len(tier_counts) >= 3:
                tier_1_pct = tier_pct.get('Tier 1 (High)', 0)
                tier_3_pct = tier_pct.get('Tier 3 (Low)', 0)
                if tier_3_pct > 0:
                    disparity_ratio = tier_1_pct / tier_3_pct
                    analysis["enrollment_concentration"]["disparity_ratio"] = round(float(disparity_ratio), 2)
                    analysis["insights"].append(
                        f"Enrollment concentration: Tier 1 districts have {disparity_ratio:.1f}x higher enrollment than Tier 3"
                    )
        
        # Urban density proxy analysis
        if 'urban_density_proxy' in self.data.columns:
            density_stats = self.data['urban_density_proxy'].describe()
            analysis["urban_density_statistics"] = {
                "mean": round(float(density_stats['mean']), 2),
                "median": round(float(density_stats['50%']), 2),
                "std": round(float(density_stats['std']), 2)
            }
        
        return analysis
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies in enrollment patterns.
        
        HACKATHON CRITERIA: Identifies anomalies
        """
        anomalies = {
            "statistical_outliers": {},
            "pattern_anomalies": {},
            "anomaly_count": 0,
            "insights": []
        }
        
        # Detect age outliers using IQR method
        if 'age' in self.data.columns:
            Q1 = self.data['age'].quantile(0.25)
            Q3 = self.data['age'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[
                (self.data['age'] < lower_bound) | (self.data['age'] > upper_bound)
            ]
            
            if len(outliers) > 0:
                anomalies["statistical_outliers"]["age_outliers"] = {
                    "count": len(outliers),
                    "percentage": round((len(outliers) / len(self.data)) * 100, 2),
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2)
                }
                anomalies["anomaly_count"] += len(outliers)
        
        # Detect temporal anomalies (unusual enrollment volumes)
        if 'enrollment_month' in self.data.columns:
            monthly_counts = self.data['enrollment_month'].value_counts()
            mean_count = monthly_counts.mean()
            std_count = monthly_counts.std()
            
            anomalous_months = monthly_counts[
                (monthly_counts > mean_count + 2 * std_count) |
                (monthly_counts < mean_count - 2 * std_count)
            ]
            
            if len(anomalous_months) > 0:
                anomalies["pattern_anomalies"]["unusual_monthly_enrollment"] = {
                    "count": len(anomalous_months),
                    "anomalous_months": anomalous_months.to_dict()
                }
                anomalies["insights"].append(
                    f"Detected {len(anomalous_months)} months with unusual enrollment volumes"
                )
        
        # Empty districts detection
        if 'district_enrollment_volume' in self.data.columns:
            low_enrollment = self.data[self.data['district_enrollment_volume'] < 10]
            if len(low_enrollment) > 0:
                unique_districts = low_enrollment['district_enrollment_volume'].nunique()
                anomalies["pattern_anomalies"]["low_enrollment_districts"] = {
                    "count": unique_districts,
                    "records": len(low_enrollment)
                }
                anomalies["insights"].append(
                    f"Identified {unique_districts} districts with very low enrollment volumes"
                )
        
        return anomalies
    
    def get_results(self) -> Dict[str, Any]:
        """Get all diagnostic analysis results."""
        return self.results


def perform_diagnostic_analysis(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Convenience function to perform diagnostic analytics.
    
    Args:
        df: Input dataframe with engineered features
        logger: Logging instance
    
    Returns:
        Dictionary with diagnostic analysis results
    """
    analyzer = DiagnosticAnalytics(logger)
    results = analyzer.analyze(df)
    
    return results
