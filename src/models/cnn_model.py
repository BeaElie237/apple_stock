#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNN1DModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        
    def build_model(self):
        """Construit le modèle CNN1D"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("✅ Modèle CNN1D construit")
        model.summary(print_fn=logger.info)
        
        return model
    
    def save_model(self, filepath='storage/models/cnn_model.h5'):
        """Sauvegarde le modèle"""
        self.model.save(filepath)
        logger.info(f"✅ Modèle sauvegardé: {filepath}")
        
    def load_model(self, filepath='storage/models/cnn_model.h5'):
        """Charge un modèle sauvegardé"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"✅ Modèle chargé: {filepath}")
        return self.model