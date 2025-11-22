# src/image_processing.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd

class FoodImageClassifier:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        
    def build_cnn_model(self, num_classes=2):
        """Build CNN using transfer learning from VGG16"""
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=(224, 224, 3))
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def prepare_image_data(self, data_path, batch_size=32):
        """Prepare image data generators"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def train_model(self, data_path, epochs=10):
        """Train the CNN model"""
        print("🖼️ Training CNN on Food Images...")
        
        train_gen, val_gen = self.prepare_image_data(data_path)
        self.model = self.build_cnn_model(num_classes=len(train_gen.class_indices))
        
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            validation_data=val_gen,
            validation_steps=val_gen.samples // val_gen.batch_size,
            epochs=epochs,
            verbose=1
        )
        
        print("✅ CNN Image Classification Training Completed")
        return history
    
    def extract_image_features(self, image_path):
        """Extract features from food images for multimodal learning"""
        if self.model is None:
            self.model = self.build_cnn_model()
            
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Extract features using intermediate layer
        feature_model = Model(inputs=self.model.input, 
                             outputs=self.model.get_layer('global_average_pooling2d').output)
        features = feature_model.predict(img_array)
        
        return features.flatten()

class MultimodalRecommender:
    """Combine image and tabular data for recommendations"""
    
    def __init__(self, tabular_model, image_model):
        self.tabular_model = tabular_model
        self.image_model = image_model
        
    def predict_recipe_quality(self, tabular_features, image_features=None):
        """Predict using both tabular and image data"""
        tabular_pred = self.tabular_model.predict(tabular_features)
        
        if image_features is not None:
            # Simple late fusion - average predictions
            image_pred = self.image_model.predict(image_features)
            combined_pred = (tabular_pred + image_pred) / 2
            return combined_pred
        
        return tabular_pred