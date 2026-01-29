#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        
    def load_model(self):
        """Charge le dernier modèle entraîné"""
        import glob
        
        if self.model_type == 'cnn':
            pattern = 'storage/models/cnn_*.h5'
        else:
            pattern = 'storage/models/gru_*.h5'
        
        model_files = glob.glob(pattern)
        if not model_files:
            raise FileNotFoundError(f"Aucun modèle {self.model_type} trouvé")
        
        latest_model = max(model_files)
        logger.info(f"📦 Chargement modèle: {latest_model}")
        
        return tf.keras.models.load_model(latest_model)
    
    def load_scaler(self):
        """Charge le scaler correspondant"""
        import joblib
        import glob
        
        scaler_files = glob.glob('storage/data/*_scaler.joblib')
        if not scaler_files:
            raise FileNotFoundError("Aucun scaler trouvé")
        
        latest_scaler = max(scaler_files)
        return joblib.load(latest_scaler)
    
    def get_last_sequence(self):
        """Récupère la dernière séquence de données"""
        import glob
        
        npz_files = glob.glob('storage/data/processed_data_*.npz')
        if not npz_files:
            raise FileNotFoundError("Aucune donnée prétraitée trouvée")
        
        latest_file = max(npz_files)
        data = np.load(latest_file)
        
        # Dernière séquence de look_back jours
        df_scaled = data['df_scaled']
        look_back = data['X_train'].shape[1]
        last_sequence = df_scaled[-look_back:, 0]
        
        return last_sequence
    
    def predict_future(self, days_to_predict=21):
        """Effectue des prédictions sur plusieurs jours"""
        logger.info(f"🔮 Prédiction sur {days_to_predict} jours...")
        
        last_sequence = self.get_last_sequence()
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(days_to_predict):
            # Prédire le prochain jour
            next_pred = self.model.predict(
                current_sequence.reshape(1, -1, 1),
                verbose=0
            )[0, 0]
            
            predictions.append(next_pred)
            
            # Mettre à jour la séquence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
            
            if (i + 1) % 5 == 0:
                logger.info(f"  Jour {i+1}/{days_to_predict} prédit")
        
        # Convertir en prix réels
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Générer les dates de prédiction
        last_date = datetime.now()
        prediction_dates = [
            last_date + timedelta(days=i+1) 
            for i in range(days_to_predict)
        ]
        
        # Créer le résultat
        result = {
            'model_type': self.model_type,
            'prediction_date': datetime.now().isoformat(),
            'predictions': predictions.tolist(),
            'prediction_dates': [d.isoformat() for d in prediction_dates],
            'first_price': float(predictions[0]),
            'last_price': float(predictions[-1]),
            'total_change': float((predictions[-1] - predictions[0]) / predictions[0] * 100)
        }
        
        logger.info(f"✅ Prédictions terminées: {result['first_price']:.2f} → {result['last_price']:.2f}")
        
        return result
    
    def save_predictions(self, predictions):
        """Sauvegarde les prédictions"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"storage/predictions/predictions_{self.model_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"💾 Prédictions sauvegardées: {filename}")
        
        # Générer un graphique
        self.plot_predictions(predictions)
        
        return filename
    
    def plot_predictions(self, predictions):
        """Génère un graphique des prédictions"""
        dates = pd.to_datetime(predictions['prediction_dates'])
        prices = predictions['predictions']
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, marker='o', linewidth=2, markersize=6)
        
        plt.title(f'Prédictions AAPL - {self.model_type.upper()} (21 jours)')
        plt.xlabel('Date')
        plt.ylabel('Prix ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Annoter le premier et dernier point
        plt.annotate(f"${prices[0]:.2f}", 
                    (dates[0], prices[0]),
                    textcoords="offset points",
                    xytext=(0,10), ha='center')
        
        plt.annotate(f"${prices[-1]:.2f}", 
                    (dates[-1], prices[-1]),
                    textcoords="offset points",
                    xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        
        plot_file = f"storage/predictions/prediction_plot_{self.model_type}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_file, dpi=100)
        plt.close()
        
        logger.info(f"📊 Graphique généré: {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'gru'], default='cnn')
    parser.add_argument('--days', type=int, default=21)
    
    args = parser.parse_args()
    
    try:
        predictor = StockPredictor(model_type=args.model)
        predictions = predictor.predict_future(days_to_predict=args.days)
        predictor.save_predictions(predictions)
        
        print("\n📋 RÉSULTATS DES PRÉDICTIONS:")
        print(f"Modèle: {args.model.upper()}")
        print(f"Premier jour: ${predictions['first_price']:.2f}")
        print(f"Dernier jour: ${predictions['last_price']:.2f}")
        print(f"Variation: {predictions['total_change']:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction: {e}")
        raise