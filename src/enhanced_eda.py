# src/enhanced_eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class EnhancedEDA:
    def __init__(self, data):
        self.data = data
        self.figures = []
        
    def create_comprehensive_visualizations(self):
        """Create all required visualizations for report"""
        self._create_feature_distributions()
        self._create_correlation_heatmap()
        self._create_missing_values_plot()
        self._create_boxplots_outliers()
        self._create_cluster_visualizations()
        
    def _create_feature_distributions(self):
        """Create distribution plots for key features"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Distributions Analysis', fontsize=16, fontweight='bold')
        
        # Nutrition Score Distribution
        axes[0,0].hist(self.data['nutrition_score'], bins=30, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.data['nutrition_score'].mean(), color='red', 
                         linestyle='--', label=f'Mean: {self.data["nutrition_score"].mean():.1f}')
        axes[0,0].set_title('Nutrition Score Distribution')
        axes[0,0].set_xlabel('Nutrition Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Price Distribution
        axes[0,1].hist(self.data['price_per_serving'], bins=30, alpha=0.7,
                      color='lightcoral', edgecolor='black')
        axes[0,1].set_title('Price per Serving Distribution')
        axes[0,1].set_xlabel('Price ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # Affordability Score Distribution
        axes[0,2].hist(self.data['affordability_score'], bins=30, alpha=0.7,
                      color='lightgreen', edgecolor='black')
        axes[0,2].set_title('Affordability Score Distribution')
        axes[0,2].set_xlabel('Affordability Score')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        # Prep Time Distribution
        axes[1,0].hist(self.data['prep_time'], bins=30, alpha=0.7,
                      color='gold', edgecolor='black')
        axes[1,0].set_title('Preparation Time Distribution')
        axes[1,0].set_xlabel('Preparation Time (min)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Time Efficiency Distribution
        axes[1,1].hist(self.data['time_efficiency'], bins=30, alpha=0.7,
                      color='violet', edgecolor='black')
        axes[1,1].set_title('Time Efficiency Distribution')
        axes[1,1].set_xlabel('Time Efficiency Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        # Popularity Distribution
        axes[1,2].hist(self.data['popularity_score'], bins=30, alpha=0.7,
                      color='orange', edgecolor='black')
        axes[1,2].set_title('Popularity Score Distribution')
        axes[1,2].set_xlabel('Popularity Score')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_correlation_heatmap(self):
        """Create comprehensive correlation heatmap"""
        numerical_cols = ['nutrition_score', 'price_per_serving', 'affordability_score',
                         'prep_time', 'time_efficiency', 'popularity_score', 'calories',
                         'protein', 'carbohydrates', 'fat', 'sugar', 'sodium']
        
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data[available_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_missing_values_plot(self):
        """Visualize missing values pattern"""
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        plt.figure(figsize=(10, 6))
        missing_percent[missing_percent > 0].plot(kind='bar', color='coral')
        plt.title('Missing Values by Feature (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots/missing_values.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_boxplots_outliers(self):
        """Create boxplots to show outliers and skewness"""
        numerical_features = ['nutrition_score', 'price_per_serving', 'prep_time', 'calories']
        available_features = [f for f in numerical_features if f in self.data.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Outlier Detection and Skewness Analysis', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(available_features):
            row, col = idx // 2, idx % 2
            self.data[feature].plot(kind='box', ax=axes[row, col], vert=True)
            axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
            axes[row, col].set_ylabel(feature.replace("_", " ").title())
            axes[row, col].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('results/plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        summary_stats = self.data[numerical_cols].describe().T
        summary_stats['skewness'] = self.data[numerical_cols].skew()
        summary_stats['missing'] = self.data[numerical_cols].isnull().sum()
        summary_stats['missing_percent'] = (summary_stats['missing'] / len(self.data)) * 100
        
        print("📊 COMPREHENSIVE STATISTICAL SUMMARY")
        print("=" * 50)
        print(summary_stats.round(3))
        
        return summary_stats