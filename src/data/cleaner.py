#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess_data(self, data):
        """Prétraite les données pour les modèles"""
        logger.info("Prétraitement des données...")
        
        # Utiliser uniquement la colonne Close
        df = data[['Close']].copy()
        
        # Normalisation
        df_scaled = self.scaler.fit_transform(df)
        
        # Création des séquences
        X, y = [], []
        for i in range(self.look_back, len(df_scaled)):
            X.append(df_scaled[i-self.look_back:i, 0])
            y.append(df_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Séparation train/test (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        logger.info(f"✅ Données prétraitées: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'df_scaled': df_scaled
        }
    
    def save_processed_data(self, processed_data, filename=None):
        """Sauvegarde les données prétraitées"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_data_{timestamp}.npz"
        
        filepath = os.path.join('storage/data', filename)
        
        # Sauvegarder les arrays numpy
        np.savez_compressed(
            filepath,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            df_scaled=processed_data['df_scaled']
        )
        
        # Sauvegarder le scaler séparément
        import joblib
        scaler_path = filepath.replace('.npz', '_scaler.joblib')
        joblib.dump(processed_data['scaler'], scaler_path)
        
        logger.info(f"✅ Données sauvegardées: {filepath}")
        
        return filepath

if __name__ == "__main__":
    # Charger les données récentes
    import glob
    latest_file = max(glob.glob('storage/data/AAPL_*.csv'))
    data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    cleaner = DataCleaner(look_back=60)
    processed = cleaner.preprocess_data(data)
    cleaner.save_processed_data(processed)