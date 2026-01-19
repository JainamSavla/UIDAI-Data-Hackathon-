"""
Visualization Module for UIDAI Hackathon Analytics Pipeline
Generates publication-quality visualizations for presentations and reports
Addresses UIDAI Judging Criteria: Visualization & Presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class Visualization:
    """
    Generate production-quality visualizations for UIDAI enrollment analysis.
    
    Key Responsibilities:
    - Create 8 publication-ready visualizations
    - Save as PNG at 300 DPI
    - Professional styling and formatting
    - Clear labeling and legends
    """
    
    def __init__(self, logger: logging.Logger, output_dir: str = "outputs"):
        """
        Initialize Visualization.
        
        Args:
            logger: Logging instance
            output_dir: Directory to save visualizations
        """
        self.logger = logger
        self.output_dir = output_dir
        
        # Set professional style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        self.visualizations_created = []
    
    def create_all_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> List[str]:
        """
        Create all 8 required visualizations.
        
        Args:
            df: Input dataframe
            analysis_results: Analysis results dictionary
        
        Returns:
            List of created visualization file paths
        """
        self.data = df
        self.analysis_results = analysis_results
        
        self.logger.info("Creating visualizations...")
        
        try:
            # 1. Age distribution chart
            self._create_age_distribution()
            
            # 2. Monthly enrollment time series
            self._create_temporal_trends()
            
            # 3. State-level choropleth (text-based heatmap as alternative)
            self._create_state_distribution()
            
            # 4. Age × Geography stacked bar
            self._create_age_geography_interaction()
            
            # 5. Time × Age multi-line chart
            self._create_time_age_trends()
            
            # 6. District clustering scatter plot
            self._create_district_clustering_visualization()
            
            # 7. Anomaly detection plot
            self._create_anomaly_visualization()
            
            # 8. Priority district intervention map
            self._create_priority_framework_visualization()
            
            self.logger.info(f"Created {len(self.visualizations_created)} visualizations")
            return self.visualizations_created
        
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _create_age_distribution(self) -> None:
        """Create age distribution chart showing age group enrollments."""
        plt.figure(figsize=(14, 8))
        
        # Check which age columns are available
        age_cols = [col for col in self.data.columns if 'age' in col.lower()]
        
        if 'demo_age_5_17' in self.data.columns and 'demo_age_17_' in self.data.columns:
            # Create grouped bar chart for age groups
            age_5_17_total = self.data['demo_age_5_17'].sum()
            age_17_plus_total = self.data['demo_age_17_'].sum()
            
            categories = ['Age 5-17', 'Age 17+']
            values = [age_5_17_total, age_17_plus_total]
            
            bars = plt.bar(categories, values, color=['steelblue', 'darkgreen'], alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add percentage labels
            total = age_5_17_total + age_17_plus_total
            for i, (cat, val) in enumerate(zip(categories, values)):
                pct = (val / total) * 100
                plt.text(i, val * 0.5, f'{pct:.1f}%', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
        
        plt.xlabel('Age Group', fontsize=12, fontweight='bold')
        plt.ylabel('Total Enrollments', fontsize=12, fontweight='bold')
        plt.title('UIDAI Enrollment Distribution by Age Group', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        self._save_figure('01_age_distribution')
    
    def _create_temporal_trends(self) -> None:
        """Create monthly enrollment time series with trend line."""
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        monthly_data = self.data.groupby(['enrollment_year', 'enrollment_month']).size().reset_index(name='count')
        monthly_data['date'] = pd.to_datetime(
            monthly_data['enrollment_year'].astype(str) + '-' + 
            monthly_data['enrollment_month'].astype(str).str.zfill(2) + '-01'
        )
        monthly_data = monthly_data.sort_values('date')
        
        # Plot
        plt.plot(monthly_data['date'], monthly_data['count'], marker='o', linewidth=2.5, 
                markersize=6, color='navy', label='Monthly Enrollment')
        
        # Add trend line
        if len(monthly_data) > 1:
            z = np.polyfit(range(len(monthly_data)), monthly_data['count'].values, 2)
            p = np.poly1d(z)
            plt.plot(monthly_data['date'], p(range(len(monthly_data))), 
                    'r--', linewidth=2, label='Trend (polynomial fit)')
        
        # Formatting
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Enrollments', fontsize=12, fontweight='bold')
        plt.title('Monthly Enrollment Trends with Temporal Pattern Analysis', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self._save_figure('02_temporal_trends')
    
    def _create_state_distribution(self) -> None:
        """Create state-level distribution visualization."""
        # Find state column
        state_col = None
        for col in self.data.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if not state_col:
            self.logger.warning("State column not found, skipping state distribution")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Top 15 states
        top_states = self.data[state_col].value_counts().head(15)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_states)))
        bars = plt.barh(range(len(top_states)), top_states.values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (idx, bar) in enumerate(zip(range(len(top_states)), bars)):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.yticks(range(len(top_states)), top_states.index, fontsize=11)
        plt.xlabel('Number of Enrollments', fontsize=12, fontweight='bold')
        plt.title('Top 15 States by Aadhaar Enrollment Volume', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        self._save_figure('03_state_distribution')
    
    def _create_age_geography_interaction(self) -> None:
        """Create age group × state stacked bar chart."""
        # Find state column
        state_col = None
        for col in self.data.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if not state_col or 'age_group' not in self.data.columns:
            self.logger.warning("Required columns not found, skipping age-geography interaction")
            return
        
        plt.figure(figsize=(16, 8))
        
        # Create crosstab
        top_states = self.data[state_col].value_counts().head(8).index
        data_subset = self.data[self.data[state_col].isin(top_states)]
        crosstab = pd.crosstab(data_subset[state_col], data_subset['age_group'])
        
        # Stacked bar chart
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), 
                     colormap='Set3', width=0.7, edgecolor='black', linewidth=0.8)
        
        plt.xlabel('State', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Enrollments', fontsize=12, fontweight='bold')
        plt.title('Age Group Distribution Across Top 8 States (Stacked Bar)', fontsize=14, fontweight='bold')
        plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self._save_figure('04_age_geography_interaction')
    
    def _create_time_age_trends(self) -> None:
        """Create multi-line chart of age group enrollment trends over time."""
        if 'enrollment_year' not in self.data.columns or 'age_group' not in self.data.columns:
            self.logger.warning("Required columns not found, skipping time-age trends")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Create pivot table - count records instead of aggregating 'age'
        pivot_data = self.data.groupby(['enrollment_year', 'age_group']).size().unstack(fill_value=0)
        
        # Multi-line chart
        for col in pivot_data.columns:
            plt.plot(pivot_data.index, pivot_data[col], marker='o', linewidth=2.5, 
                    markersize=8, label=str(col))
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Enrollments', fontsize=12, fontweight='bold')
        plt.title('Enrollment Trends by Age Group Over Time (Lifecycle Analysis)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, title='Age Group')
        plt.grid(True, alpha=0.3)
        plt.xticks(pivot_data.index, rotation=45)
        
        self._save_figure('05_time_age_trends')
    
    def _create_district_clustering_visualization(self) -> None:
        """Create district clustering scatter plot."""
        if 'district_enrollment_volume' not in self.data.columns:
            self.logger.warning("District clustering data not found")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Prepare district data
        district_col = None
        for col in self.data.columns:
            if 'district' in col.lower():
                district_col = col
                break
        
        if not district_col:
            return
        
        district_stats = []
        for district in self.data[district_col].unique():
            district_data = self.data[self.data[district_col] == district]
            stats = {
                'district': district,
                'enrollment_volume': len(district_data),
                'avg_age': district_data['age'].mean() if 'age' in self.data.columns else 0,
                'vulnerability_ratio': (
                    district_data['is_vulnerable_group'].mean() * 100 
                    if 'is_vulnerable_group' in self.data.columns else 0
                )
            }
            district_stats.append(stats)
        
        districts_df = pd.DataFrame(district_stats)
        
        # Scatter plot
        scatter = plt.scatter(districts_df['enrollment_volume'], 
                            districts_df['avg_age'],
                            s=districts_df['vulnerability_ratio']*20,
                            c=districts_df['vulnerability_ratio'],
                            cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Vulnerability Ratio (%)', fontweight='bold')
        
        plt.xlabel('Enrollment Volume', fontsize=12, fontweight='bold')
        plt.ylabel('Average Age (years)', fontsize=12, fontweight='bold')
        plt.title('District Clustering: Enrollment Volume vs Demographics\n(Bubble size = Vulnerability Ratio)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        self._save_figure('06_district_clustering')
    
    def _create_anomaly_visualization(self) -> None:
        """Create anomaly detection visualization."""
        if 'enrollment_month' not in self.data.columns:
            self.logger.warning("Required data for anomaly visualization not found")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Monthly enrollment data
        monthly_counts = self.data['enrollment_month'].value_counts().sort_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Calculate mean and std for anomaly detection
        mean_val = monthly_counts.mean()
        std_val = monthly_counts.std()
        upper_bound = mean_val + 2 * std_val
        lower_bound = mean_val - 2 * std_val
        
        # Plot
        colors = ['red' if v > upper_bound or v < lower_bound else 'steelblue' 
                 for v in monthly_counts.values]
        plt.bar(range(len(monthly_counts)), monthly_counts.values, color=colors, 
               edgecolor='black', linewidth=1.2, alpha=0.7)
        
        # Add bounds
        plt.axhline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.0f}')
        plt.axhline(upper_bound, color='red', linestyle='--', linewidth=2, label=f'2σ Upper Bound')
        plt.axhline(lower_bound, color='red', linestyle='--', linewidth=2, label=f'2σ Lower Bound')
        
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Enrollments', fontsize=12, fontweight='bold')
        plt.title('Anomaly Detection: Monthly Enrollment with Statistical Bounds', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(monthly_counts)), 
                  [months[i-1] if i <= len(months) else f'M{i}' for i in monthly_counts.index])
        plt.legend(loc='best', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        self._save_figure('07_anomaly_detection')
    
    def _create_priority_framework_visualization(self) -> None:
        """Create priority district intervention framework visualization."""
        if 'district_priority_tier' not in self.data.columns:
            self.logger.warning("District priority tier data not found")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Tier distribution
        tier_counts = self.data['district_priority_tier'].value_counts().sort_index()
        
        # Custom colors for tiers
        colors_map = {
            'Tier 1 (High)': '#d62728',     # Red
            'Tier 2 (Medium)': '#ff7f0e',   # Orange
            'Tier 3 (Low)': '#2ca02c'       # Green
        }
        colors = [colors_map.get(str(t), 'steelblue') for t in tier_counts.index]
        
        # Pie chart with details
        wedges, texts, autotexts = plt.pie(
            tier_counts.values, 
            labels=tier_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2}
        )
        
        # Enhance autotext
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        plt.title('District Priority Framework for Enrollment Intervention', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with intervention details
        legend_labels = [
            'Tier 1: Critical intervention (immediate)',
            'Tier 2: High priority intervention',
            'Tier 3: Optimization & monitoring'
        ]
        plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        self._save_figure('08_priority_framework')
    
    def _save_figure(self, filename: str) -> None:
        """Save figure with high quality settings."""
        filepath = f"{self.output_dir}/{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        self.visualizations_created.append(filepath)
        self.logger.info(f"Saved visualization: {filepath}")
    
    def get_created_visualizations(self) -> List[str]:
        """Get list of created visualizations."""
        return self.visualizations_created


def create_visualizations(df: pd.DataFrame, analysis_results: Dict, 
                         logger: logging.Logger, output_dir: str = "outputs") -> List[str]:
    """
    Convenience function to create all visualizations.
    
    Args:
        df: Input dataframe
        analysis_results: Analysis results dictionary
        logger: Logging instance
        output_dir: Output directory for visualizations
    
    Returns:
        List of created visualization file paths
    """
    visualizer = Visualization(logger, output_dir)
    return visualizer.create_all_visualizations(df, analysis_results)
