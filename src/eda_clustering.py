# src/eda_clustering.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import os
import traceback
import sys
from datetime import datetime

class EDAClustering:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.clustering_results = {}
        self.error_log = []
        
    def _log_error(self, error_type, message, details=""):
        """Log errors with timestamp and details"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': message,
            'details': details
        }
        self.error_log.append(error_entry)
        print(f"❌ ERROR [{timestamp}] {error_type}: {message}")
        if details:
            print(f"   Details: {details}")
    
    def _safe_execute(self, func, operation_name, fallback_value=None):
        """Safe execution wrapper with error handling"""
        try:
            print(f"🔄 Executing: {operation_name}")
            return func()
        except MemoryError as e:
            self._log_error("MemoryError", f"Memory issue in {operation_name}", str(e))
            return fallback_value
        except ValueError as e:
            self._log_error("ValueError", f"Data value issue in {operation_name}", str(e))
            return fallback_value
        except Exception as e:
            self._log_error("RuntimeError", f"Unexpected error in {operation_name}", 
                          f"{str(e)}\n{traceback.format_exc()}")
            return fallback_value
    
    def _validate_data(self):
        """Validate data before processing"""
        print("🔍 Validating dataset...")
        
        if self.data is None or self.data.empty:
            self._log_error("DataError", "Dataset is empty or None")
            return False
        
        print(f"📊 Dataset shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        
        # Check for required numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            self._log_error("DataError", f"Insufficient numerical columns. Found: {len(numerical_cols)}")
            return False
        
        # Check for NaN values
        nan_counts = self.data[numerical_cols].isna().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  NaN values found: {dict(nan_counts[nan_counts > 0])}")
            # Fill NaN with mean for numerical columns
            self.data[numerical_cols] = self.data[numerical_cols].fillna(self.data[numerical_cols].mean())
            print("✅ NaN values filled with column means")
        
        # Check for infinite values
        inf_mask = np.isinf(self.data[numerical_cols])
        if inf_mask.any().any():
            print(f"⚠️  Infinite values found, replacing with max/min values")
            for col in numerical_cols:
                col_data = self.data[col]
                max_val = col_data[np.isfinite(col_data)].max()
                min_val = col_data[np.isfinite(col_data)].min()
                self.data[col] = col_data.replace([np.inf, -np.inf], [max_val, min_val])
        
        return True
    
    def comprehensive_eda(self):
        """Perform comprehensive exploratory data analysis (R2 Requirement)"""
        print("📊 Performing comprehensive EDA...")
        
        if not self._validate_data():
            self._log_error("EDAAError", "Data validation failed, skipping EDA")
            return None
        
        def create_plots():
            # Set up plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('NutriBudget AI - Exploratory Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. Nutrition Score Distribution
            if 'nutrition_score' in self.data.columns:
                axes[0,0].hist(self.data['nutrition_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0,0].axvline(self.data['nutrition_score'].mean(), color='red', linestyle='--', 
                                label=f'Mean: {self.data["nutrition_score"].mean():.1f}')
                axes[0,0].set_title('Distribution of Nutrition Scores')
                axes[0,0].set_xlabel('Nutrition Score')
                axes[0,0].set_ylabel('Frequency')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
            
            # 2. Price vs Nutrition Scatter
            if 'price_per_serving' in self.data.columns and 'nutrition_score' in self.data.columns:
                affordability_color = self.data.get('affordability_score', 0.5)
                if isinstance(affordability_color, (pd.Series, np.ndarray)):
                    scatter = axes[0,1].scatter(self.data['price_per_serving'], self.data['nutrition_score'], 
                                 alpha=0.6, c=affordability_color, cmap='viridis', s=50)
                    plt.colorbar(scatter, ax=axes[0,1])
                axes[0,1].set_title('Price vs Nutrition Score')
                axes[0,1].set_xlabel('Price per Meal ($)')
                axes[0,1].set_ylabel('Nutrition Score')
                axes[0,1].grid(True, alpha=0.3)
            
            # 3. Correlation Heatmap
            numerical_cols = ['nutrition_score', 'price_per_serving', 'affordability_score', 
                             'prep_time', 'time_efficiency', 'popularity_score']
            available_cols = [col for col in numerical_cols if col in self.data.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = self.data[available_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,2], fmt='.2f')
                axes[0,2].set_title('Feature Correlation Heatmap')
            else:
                axes[0,2].text(0.5, 0.5, 'Insufficient data\nfor correlation matrix', 
                              ha='center', va='center', transform=axes[0,2].transAxes)
                axes[0,2].set_title('Feature Correlation Heatmap')
            
            # 4. Time Efficiency Distribution
            if 'time_efficiency' in self.data.columns:
                axes[1,0].hist(self.data['time_efficiency'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1,0].set_title('Distribution of Time Efficiency')
                axes[1,0].set_xlabel('Time Efficiency Score')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].grid(True, alpha=0.3)
            
            # 5. Affordability Score Distribution
            if 'affordability_score' in self.data.columns:
                axes[1,1].hist(self.data['affordability_score'], bins=30, alpha=0.7, color='orange', edgecolor='black')
                axes[1,1].set_title('Distribution of Affordability Scores')
                axes[1,1].set_xlabel('Affordability Score')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].grid(True, alpha=0.3)
            
            # 6. Popularity vs Nutrition
            if 'nutrition_score' in self.data.columns and 'popularity_score' in self.data.columns:
                price_color = self.data.get('price_per_serving', 0.5)
                if isinstance(price_color, (pd.Series, np.ndarray)):
                    scatter2 = axes[1,2].scatter(self.data['nutrition_score'], self.data['popularity_score'], 
                                 alpha=0.6, c=price_color, cmap='plasma', s=50)
                    plt.colorbar(scatter2, ax=axes[1,2])
                axes[1,2].set_title('Nutrition vs Popularity')
                axes[1,2].set_xlabel('Nutrition Score')
                axes[1,2].set_ylabel('Popularity Score')
                axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.makedirs(os.path.join(base_dir, 'results', 'plots'), exist_ok=True)
            plot_path = os.path.join(base_dir, 'results', 'plots', 'comprehensive_eda.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return True
        
        result = self._safe_execute(create_plots, "EDA Plot Creation", False)
        
        # Statistical summary
        if result:
            print("\n📈 STATISTICAL SUMMARY:")
            numerical_cols = ['nutrition_score', 'price_per_serving', 'affordability_score', 
                             'prep_time', 'time_efficiency', 'popularity_score']
            summary_cols = [col for col in numerical_cols if col in self.data.columns]
            if summary_cols:
                print(self.data[summary_cols].describe())
                
                # Correlation insights
                if len(summary_cols) >= 2:
                    corr_matrix = self.data[summary_cols].corr()
                    highest_corr = corr_matrix.unstack().sort_values(ascending=False)
                    unique_corrs = highest_corr[highest_corr < 0.99]  # Remove self-correlations
                    if len(unique_corrs) > 0:
                        print(f"\n🔗 Highest correlation: {unique_corrs.index[0]} = {unique_corrs.iloc[0]:.3f}")
        
        return result
    
    def _safe_clustering_operation(self, operation_func, operation_name, fallback_value=None):
        """Execute clustering operations with memory and error protection"""
        try:
            return operation_func()
        except MemoryError as e:
            self._log_error("MemoryError", f"Memory exhausted during {operation_name}", str(e))
            return fallback_value
        except ValueError as e:
            self._log_error("ValueError", f"Invalid data for {operation_name}", str(e))
            return fallback_value
        except Exception as e:
            self._log_error("ClusteringError", f"Unexpected error in {operation_name}", 
                          f"{str(e)}\n{traceback.format_exc()}")
            return fallback_value
    
    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering with optimal cluster selection (R2 Requirement)"""
        print("🔍 Performing clustering analysis...")
        
        # Validate data first
        if not self._validate_data():
            self._log_error("ClusteringError", "Data validation failed, cannot perform clustering")
            self.data['cluster'] = 0
            return self.data
        
        # Features for clustering
        cluster_features = ['nutrition_score', 'affordability_score', 'time_efficiency', 'popularity_score']
        available_features = [f for f in cluster_features if f in self.data.columns]
        
        if len(available_features) < 2:
            print("⚠️ Insufficient features for clustering. Using available numerical features.")
            available_features = [col for col in self.data.select_dtypes(include=[np.number]).columns 
                                if col not in ['recipe_id', 'cluster']][:4]
            if len(available_features) < 2:
                self._log_error("ClusteringError", 
                              f"Only {len(available_features)} numerical features available. Need at least 2.")
                self.data['cluster'] = 0
                return self.data
        
        print(f"📊 Clustering features: {available_features}")
        print(f"📊 Data shape for clustering: {self.data[available_features].shape}")
        
        X = self.data[available_features].copy()
        
        # Data preprocessing with error handling
        def preprocess_data():
            # Handle NaN and infinite values
            X_clean = X.fillna(X.mean())
            for col in X_clean.columns:
                col_data = X_clean[col]
                if np.isinf(col_data).any():
                    finite_vals = col_data[np.isfinite(col_data)]
                    if len(finite_vals) > 0:
                        max_val = finite_vals.max()
                        min_val = finite_vals.min()
                        X_clean[col] = col_data.replace([np.inf, -np.inf], [max_val, min_val])
                    else:
                        X_clean[col] = 0
            
            # Check for constant columns
            constant_cols = []
            for col in X_clean.columns:
                if X_clean[col].nunique() <= 1:
                    constant_cols.append(col)
                    print(f"⚠️  Constant column detected: {col}")
            
            if constant_cols:
                print(f"🔄 Removing constant columns: {constant_cols}")
                X_clean = X_clean.drop(columns=constant_cols)
                available_features_clean = [f for f in available_features if f not in constant_cols]
            else:
                available_features_clean = available_features
            
            if len(available_features_clean) < 2:
                raise ValueError(f"Only {len(available_features_clean)} non-constant features remaining")
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # Check for NaN in scaled data
            if np.isnan(X_scaled).any():
                raise ValueError("NaN values found in scaled data")
            
            return X_scaled, available_features_clean
        
        preprocessing_result = self._safe_clustering_operation(preprocess_data, "Data Preprocessing")
        if preprocessing_result is None:
            self._log_error("ClusteringError", "Data preprocessing failed")
            self.data['cluster'] = 0
            return self.data
        
        X_scaled, available_features_clean = preprocessing_result
        
        # Find optimal number of clusters
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, min(8, len(X_scaled) // 10))
        
        print(f"🔍 Testing cluster range: {list(k_range)}")
        
        for k in k_range:
            if k >= len(X_scaled):
                print(f"⚠️  Skipping k={k} (too many clusters for dataset size)")
                break
                
            print(f"  Testing {k} clusters...")
            
            def fit_kmeans(k_value):
                kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10, max_iter=100)
                labels = kmeans.fit_predict(X_scaled)
                return labels, kmeans
            
            kmeans_result = self._safe_clustering_operation(
                lambda: fit_kmeans(k), f"KMeans with k={k}"
            )
            
            if kmeans_result is None:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                continue
                
            labels, kmeans = kmeans_result
            
            # Calculate metrics safely
            def calculate_metrics():
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    cal_score = calinski_harabasz_score(X_scaled, labels)
                    return sil_score, cal_score
                else:
                    return 0, 0
            
            metrics = self._safe_clustering_operation(calculate_metrics, f"Metrics calculation k={k}", (0, 0))
            sil_score, cal_score = metrics
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            print(f"    Silhouette: {sil_score:.4f}, Calinski: {cal_score:.4f}")
        
        # Check if we have valid scores
        if not silhouette_scores or max(silhouette_scores) <= 0:
            self._log_error("ClusteringError", "No valid clustering results obtained")
            self.data['cluster'] = 0
            return self.data
        
        # Plot cluster evaluation
        def create_cluster_eval_plots():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Cluster Analysis - Optimal K Selection', fontsize=14, fontweight='bold')
            
            ax1.plot(list(k_range)[:len(silhouette_scores)], silhouette_scores, 'bo-', linewidth=2, markersize=8)
            ax1.set_title('Silhouette Scores for Different K')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(list(k_range)[:len(calinski_scores)], calinski_scores, 'ro-', linewidth=2, markersize=8)
            ax2.set_title('Calinski-Harabasz Scores for Different K')
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Calinski-Harabasz Score')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.makedirs(os.path.join(base_dir, 'results', 'plots'), exist_ok=True)
            plot_path = os.path.join(base_dir, 'results', 'plots', 'cluster_evaluation.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            return True
        
        self._safe_execute(create_cluster_eval_plots, "Cluster Evaluation Plots", False)
        
        # Choose optimal k (max silhouette score)
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"✅ Optimal number of clusters: {optimal_k}")
        
        # Final clustering with optimal k
        def final_clustering():
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=200)
            cluster_labels = final_kmeans.fit_predict(X_scaled)
            self.data['cluster'] = cluster_labels
            
            # PCA for visualization
            if len(available_features_clean) >= 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                self.data['pca1'] = X_pca[:, 0]
                self.data['pca2'] = X_pca[:, 1]
                
                # Visualize clusters
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                    cmap='viridis', alpha=0.7, s=50)
                plt.colorbar(scatter, label='Cluster')
                plt.title(f'Recipe Clusters (K={optimal_k}) - PCA Visualization', fontweight='bold')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                
                # Add cluster centers
                centers_pca = pca.transform(final_kmeans.cluster_centers_)
                plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, 
                           label='Cluster Centers', edgecolors='black')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(base_dir, 'results', 'plots', 'cluster_visualization.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.show()
            
            return final_kmeans, cluster_labels
        
        final_result = self._safe_clustering_operation(final_clustering, "Final Clustering")
        if final_result is None:
            self._log_error("ClusteringError", "Final clustering failed, using single cluster")
            self.data['cluster'] = 0
            final_kmeans = None
            cluster_labels = np.zeros(len(self.data))
        else:
            final_kmeans, cluster_labels = final_result
        
        # Analyze cluster characteristics
        def analyze_clusters():
            cluster_summary = self.data.groupby('cluster')[available_features_clean].mean()
            cluster_summary['count'] = self.data.groupby('cluster').size()
            cluster_summary['percentage'] = (cluster_summary['count'] / len(self.data) * 100).round(1)
            
            print("\n📊 CLUSTER CHARACTERISTICS:")
            print(cluster_summary)
            return cluster_summary
        
        cluster_summary = self._safe_execute(analyze_clusters, "Cluster Analysis", pd.DataFrame())
        
        self.clustering_results = {
            'model': final_kmeans,
            'labels': cluster_labels,
            'optimal_k': optimal_k,
            'silhouette_score': max(silhouette_scores) if silhouette_scores else 0,
            'cluster_summary': cluster_summary,
            'error_log': self.error_log
        }
        
        print(f"✅ Clustering completed with silhouette score: {max(silhouette_scores):.3f}")
        
        # Print error summary if any
        if self.error_log:
            print(f"\n⚠️  Completed with {len(self.error_log)} warnings/errors")
            for error in self.error_log[-3:]:  # Show last 3 errors
                print(f"   - {error['type']}: {error['message']}")
        
        return self.data
    
    def get_error_report(self):
        """Generate comprehensive error report"""
        if not self.error_log:
            return "✅ No errors encountered"
        
        report = f"📊 ERROR REPORT ({len(self.error_log)} entries)\n"
        report += "=" * 50 + "\n"
        
        for i, error in enumerate(self.error_log, 1):
            report += f"{i}. [{error['timestamp']}] {error['type']}\n"
            report += f"   Message: {error['message']}\n"
            if error['details']:
                report += f"   Details: {error['details'][:100]}...\n"
            report += "\n"
        
        return report