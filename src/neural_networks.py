# src/neural_networks.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

class NeuralNetworks:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = StandardScaler()
        self.models = {}
        self.histories = {}
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
    def build_mlp(self, input_dim):
        """Build Multi-Layer Perceptron (R4 Requirement)"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn(self, input_dim):
        """Build 1D CNN for structured data (R4 Requirement)"""
        # Reshape data for CNN (samples, timesteps, features)
        X_train_reshaped = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], input_dim, 1)
        X_test_reshaped = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], input_dim, 1)
        
        # Use simpler architecture that works with any input size
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(input_dim, 1), padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model, X_train_reshaped, X_test_reshaped
    
    def train_mlp(self):
        """Train MLP model (R4 Requirement)"""
        print("🧠 Training MLP...")
        
        input_dim = self.X_train_scaled.shape[1]
        mlp_model = self.build_mlp(input_dim)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_loss')
        ]
        
        print(f"📊 MLP Architecture: Input dim={input_dim}, 4 hidden layers")
        
        history = mlp_model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_test_scaled, self.y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['MLP'] = mlp_model
        self.histories['MLP'] = history
        
        print("✅ MLP training completed successfully")
        return mlp_model, history
    
    def train_cnn(self):
        """Train CNN model (R4 Requirement)"""
        print("🔄 Training CNN...")
        
        input_dim = self.X_train_scaled.shape[1]
        
        # Ensure input dimension is suitable for CNN
        if input_dim < 3:
            print(f"⚠️ Input dimension {input_dim} too small for CNN. Using MLP instead.")
            return self.train_mlp()
        
        cnn_model, X_train_reshaped, X_test_reshaped = self.build_cnn(input_dim)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_loss')
        ]
        
        print(f"📊 CNN Architecture: Input dim={input_dim}, 2 Conv1D layers + 3 Dense layers")
        
        history = cnn_model.fit(
            X_train_reshaped, self.y_train,
            validation_data=(X_test_reshaped, self.y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['CNN'] = cnn_model
        self.histories['CNN'] = history
        self.cnn_data = (X_train_reshaped, X_test_reshaped)
        
        print("✅ CNN training completed successfully")
        return cnn_model, history
    
    def evaluate_neural_network(self, model, name, X_test=None):
        """Evaluate neural network performance (R4 Requirement)"""
        if X_test is None:
            X_test = self.X_test_scaled
        
        try:
            # Predictions
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            # Handle AUC calculation safely
            try:
                auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            except:
                auc_roc = 0.5  # Random classifier
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n📊 {name} Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   AUC-ROC: {auc_roc:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            return None
    
    def plot_training_history(self):
        """Plot training history for all neural networks"""
        if not self.histories:
            print("⚠️ No training history available.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Networks Training History', fontsize=16, fontweight='bold')
        
        for idx, (name, history) in enumerate(self.histories.items()):
            # Accuracy plot
            if 'accuracy' in history.history:
                axes[0,0].plot(history.history['accuracy'], label=f'{name} Train', alpha=0.7)
                axes[0,0].plot(history.history['val_accuracy'], label=f'{name} Val', linestyle='--', alpha=0.7)
                axes[0,0].set_title('Model Accuracy')
                axes[0,0].set_ylabel('Accuracy')
                axes[0,0].set_xlabel('Epoch')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
            
            # Loss plot
            if 'loss' in history.history:
                axes[0,1].plot(history.history['loss'], label=f'{name} Train', alpha=0.7)
                axes[0,1].plot(history.history['val_loss'], label=f'{name} Val', linestyle='--', alpha=0.7)
                axes[0,1].set_title('Model Loss')
                axes[0,1].set_ylabel('Loss')
                axes[0,1].set_xlabel('Epoch')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
            
            # Precision plot
            if 'precision' in history.history:
                axes[1,0].plot(history.history['precision'], label=f'{name} Train', alpha=0.7)
                axes[1,0].plot(history.history['val_precision'], label=f'{name} Val', linestyle='--', alpha=0.7)
                axes[1,0].set_title('Model Precision')
                axes[1,0].set_ylabel('Precision')
                axes[1,0].set_xlabel('Epoch')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
            
            # Recall plot
            if 'recall' in history.history:
                axes[1,1].plot(history.history['recall'], label=f'{name} Train', alpha=0.7)
                axes[1,1].plot(history.history['val_recall'], label=f'{name} Val', linestyle='--', alpha=0.7)
                axes[1,1].set_title('Model Recall')
                axes[1,1].set_ylabel('Recall')
                axes[1,1].set_xlabel('Epoch')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_path = os.path.join(base_dir, 'results', 'plots', 'neural_networks_training.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_all_neural_networks(self):
        """Train all neural network architectures (R4 Requirement)"""
        print("="*50)
        print("TRAINING NEURAL NETWORKS (R4 REQUIREMENT)")
        print("="*50)
        
        nn_results = {}
        
        try:
            self.train_mlp()
        except Exception as e:
            print(f"❌ MLP training failed: {e}")
        
        try:
            self.train_cnn()
        except Exception as e:
            print(f"❌ CNN training failed: {e}")
        
        # Evaluate all models
        print("\n📈 Evaluating neural networks...")
        for name, model in self.models.items():
            try:
                if name == 'CNN' and hasattr(self, 'cnn_data'):
                    X_test_cnn = self.cnn_data[1]
                    results = self.evaluate_neural_network(model, name, X_test_cnn)
                else:
                    results = self.evaluate_neural_network(model, name)
                
                if results:
                    nn_results[name] = results
            except Exception as e:
                print(f"❌ Evaluation failed for {name}: {e}")
        
        if self.histories:
            self.plot_training_history()
        
        print(f"✅ Trained and evaluated {len(nn_results)} neural networks")
        return nn_results