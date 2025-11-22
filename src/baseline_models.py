# src/baseline_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os

class BaselineModels:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def prepare_data(self):
        """Prepare data for classification (R3 Requirement)"""
        print("📊 Preparing data for baseline models...")
        
        # Create target variable: High-quality affordable recipes (1) vs others (0)
        if 'nutrition_score' in self.data.columns and 'affordability_score' in self.data.columns:
            nutrition_threshold = self.data['nutrition_score'].quantile(0.7)
            affordability_threshold = self.data['affordability_score'].quantile(0.7)
            
            self.data['target'] = ((self.data['nutrition_score'] >= nutrition_threshold) & 
                                  (self.data['affordability_score'] >= affordability_threshold)).astype(int)
            
            print(f"🎯 Target distribution:")
            print(f"   High-quality recipes: {self.data['target'].sum()} ({self.data['target'].mean()*100:.1f}%)")
            print(f"   Other recipes: {len(self.data) - self.data['target'].sum()} ({(1-self.data['target'].mean())*100:.1f}%)")
        else:
            # Fallback target creation
            print("⚠️ Using fallback target creation")
            self.data['target'] = (np.random.random(len(self.data)) > 0.7).astype(int)
        
        # Features for modeling - using available real features
        possible_features = ['nutrition_score', 'affordability_score', 'time_efficiency', 
                           'popularity_score', 'prep_time', 'price_per_serving', 'calories', 
                           'protein', 'carbohydrates', 'fat', 'sugar', 'sodium', 'difficulty_encoded',
                           'ingredient_count', 'step_count', 'complexity_score']
        
        features = [f for f in possible_features if f in self.data.columns]
        
        if len(features) < 3:
            # Add any numerical features
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [col for col in numerical_cols if col != 'target' and col != 'recipe_id'][:8]
        
        X = self.data[features]
        y = self.data['target']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"✅ Data preparation completed:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        print(f"   Features used: {features}")
        print(f"   Positive class ratio: {y.mean():.3f}")
        
    def train_decision_tree(self):
        """Train and optimize Decision Tree (R3 Requirement)"""
        print("🌳 Training Decision Tree...")
        
        try:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            best_dt = grid_search.best_estimator_
            self.models['Decision Tree'] = best_dt
            
            print(f"✅ Decision Tree trained:")
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return best_dt
            
        except Exception as e:
            print(f"❌ Decision Tree training failed: {e}. Using default parameters.")
            dt = DecisionTreeClassifier(random_state=42, max_depth=5)
            dt.fit(self.X_train, self.y_train)
            self.models['Decision Tree'] = dt
            return dt
    
    def train_naive_bayes(self):
        """Train Naive Bayes classifier (R3 Requirement)"""
        print("📊 Training Naive Bayes...")
        
        try:
            nb = GaussianNB()
            nb.fit(self.X_train, self.y_train)
            self.models['Naive Bayes'] = nb
            print("✅ Naive Bayes trained successfully")
            return nb
        except Exception as e:
            print(f"❌ Naive Bayes training failed: {e}")
            return None
    
    def train_logistic_regression(self):
        """Train Logistic Regression with regularization (R3 Requirement)"""
        print("📈 Training Logistic Regression...")
        
        try:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2'],  # Simplified for stability
                'solver': ['liblinear']
            }
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            best_lr = grid_search.best_estimator_
            self.models['Logistic Regression'] = best_lr
            
            print(f"✅ Logistic Regression trained:")
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return best_lr
            
        except Exception as e:
            print(f"❌ Logistic Regression training failed: {e}. Using default parameters.")
            lr = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
            lr.fit(self.X_train, self.y_train)
            self.models['Logistic Regression'] = lr
            return lr
    
    def train_knn(self):
        """Train k-Nearest Neighbors (R3 Requirement)"""
        print("🎯 Training k-NN...")
        
        try:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            best_knn = grid_search.best_estimator_
            self.models['k-NN'] = best_knn
            
            print(f"✅ k-NN trained:")
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return best_knn
            
        except Exception as e:
            print(f"❌ k-NN training failed: {e}. Using default parameters.")
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(self.X_train, self.y_train)
            self.models['k-NN'] = knn
            return knn
    
    def train_random_forest(self):
        """Train Random Forest (R3 Requirement)"""
        print("🌲 Training Random Forest...")
        
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = rf
            print("✅ Random Forest trained successfully")
            return rf
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
            return None
    
    def train_svm(self):
        """Train Support Vector Machine (Additional model)"""
        print("⚡ Training SVM...")
        
        try:
            svm = SVC(probability=True, random_state=42)
            svm.fit(self.X_train, self.y_train)
            self.models['SVM'] = svm
            print("✅ SVM trained successfully")
            return svm
        except Exception as e:
            print(f"❌ SVM training failed: {e}")
            return None
    
    def evaluate_model(self, model, name):
        """Comprehensive model evaluation (R3 Requirement)"""
        if model is None:
            print(f"⚠️ Model {name} is None, skipping evaluation")
            return None
            
        try:
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics with error handling
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            # Get probabilities if available
            y_pred_proba = None
            auc_roc = 0
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                try:
                    auc_roc = roc_auc_score(self.y_test, y_pred_proba)
                except:
                    auc_roc = 0
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = f1
                cv_std = 0
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_roc': auc_roc
            }
            
            self.results[name] = results
            
            print(f"\n📊 {name} Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            if auc_roc > 0:
                print(f"   AUC-ROC: {auc_roc:.4f}")
            print(f"   CV F1: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating model {name}: {e}")
            return None
    
    def train_all_models(self):
        """Train all baseline models (R3 Requirement)"""
        print("="*50)
        print("TRAINING BASELINE MODELS (R3 REQUIREMENT)")
        print("="*50)
        
        self.prepare_data()
        
        # Train individual models (5+ algorithms as required)
        print("\n🚀 Training 5+ machine learning algorithms...")
        self.train_decision_tree()
        self.train_naive_bayes()
        self.train_logistic_regression()
        self.train_knn()
        self.train_random_forest()
        self.train_svm()  # Additional model
        
        # Evaluate all models
        print("\n📈 Evaluating all models...")
        for name, model in self.models.items():
            self.evaluate_model(model, name)
        
        print(f"✅ Trained and evaluated {len(self.models)} baseline models")
    
    def comparative_analysis(self):
        """Compare all models and create visualizations (R3 Requirement)"""
        print("\n" + "="*50)
        print("COMPARATIVE ANALYSIS OF BASELINE MODELS")
        print("="*50)
        
        if not self.results:
            print("❌ No results to compare")
            return pd.DataFrame()
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        
        # Ensure numeric types
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean', 'auc_roc']
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Plot performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Models Comparative Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Bar plot of main metrics
            available_metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1_score'] 
                               if m in results_df.columns]
            
            if available_metrics:
                results_df[available_metrics].plot(kind='bar', ax=axes[0,0], colormap='Set3')
                axes[0,0].set_title('Model Performance Comparison')
                axes[0,0].set_ylabel('Score')
                axes[0,0].tick_params(axis='x', rotation=45)
                axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0,0].grid(True, alpha=0.3)
            
            # Cross-validation performance
            if 'cv_mean' in results_df.columns and 'cv_std' in results_df.columns:
                cv_means = results_df['cv_mean'].values
                cv_stds = results_df['cv_std'].values
                models = results_df.index.tolist()
                
                bars = axes[0,1].bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='lightblue')
                axes[0,1].set_title('Cross-Validation F1 Scores')
                axes[0,1].set_ylabel('F1 Score')
                axes[0,1].tick_params(axis='x', rotation=45)
                axes[0,1].grid(True, alpha=0.3)
            
            # Confusion matrices for top 2 models
            if 'f1_score' in results_df.columns:
                valid_results = [(name, res) for name, res in self.results.items() 
                               if res and 'f1_score' in res and 'predictions' in res]
                
                if valid_results:
                    top_models = sorted(valid_results, key=lambda x: x[1]['f1_score'], reverse=True)[:2]
                    
                    for idx, (name, results) in enumerate(top_models):
                        cm = confusion_matrix(self.y_test, results['predictions'])
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, idx])
                        axes[1, idx].set_title(f'Confusion Matrix - {name}')
                        axes[1, idx].set_xlabel('Predicted')
                        axes[1, idx].set_ylabel('Actual')
            
            # ROC Curves for models with probabilities
            models_with_proba = [(name, res) for name, res in self.results.items() 
                               if res and 'probabilities' in res and res['probabilities'] is not None]
            
            if models_with_proba:
                for name, results in models_with_proba:
                    fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
                    auc_score = results.get('auc_roc', 0)
                    axes[1,1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                
                axes[1,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                axes[1,1].set_xlabel('False Positive Rate')
                axes[1,1].set_ylabel('True Positive Rate')
                axes[1,1].set_title('ROC Curves')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plot_path = os.path.join(base_dir, 'results', 'plots', 'baseline_models_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print comprehensive comparison
            print("\n🏆 RANKED MODEL PERFORMANCE:")
            if 'f1_score' in results_df.columns:
                ranked_models = results_df.sort_values('f1_score', ascending=False)
                
                for rank, (name, scores) in enumerate(ranked_models.iterrows(), 1):
                    print(f"   {rank}. {name:20} | F1 = {scores['f1_score']:.4f} | "
                          f"Accuracy = {scores.get('accuracy', 0):.4f} | "
                          f"AUC = {scores.get('auc_roc', 0):.4f}")
            else:
                print("❌ No F1 scores available for ranking")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error in comparative analysis: {e}")
            return results_df