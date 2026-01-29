#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import logging
import argparse

from cnn_model import CNN1DModel
from gru_model import GRUModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type='both'):
        self.model_type = model_type
        self.models = {}
        self.histories = {}
        
    def load_data(self):
        """Charge les données prétraitées"""
        import glob
        import joblib
        
        # Trouver le fichier le plus récent
        npz_files = glob.glob('storage/data/processed_data_*.npz')
        if not npz_files:
            raise FileNotFoundError("Aucune donnée prétraitées trouvée")
        
        latest_file = max(npz_files)
        scaler_file = latest_file.replace('.npz', '_scaler.joblib')
        
        # Charger les données
        data = np.load(latest_file)
        scaler = joblib.load(scaler_file)
        
        logger.info(f"📊 Données chargées: {latest_file}")
        
        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'df_scaled': data['df_scaled'],
            'scaler': scaler
        }
    
    def train_cnn(self, data):
        """Entraîne le modèle CNN1D"""
        logger.info("🏋️  Entraînement CNN1D...")
        
        cnn = CNN1DModel(input_shape=(data['X_train'].shape[1], 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'storage/models/cnn_best.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Entraînement
        history = cnn.model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_test'], data['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder
        cnn.save_model(f"storage/models/cnn_{datetime.now().strftime('%Y%m%d')}.h5")
        
        self.models['cnn'] = cnn
        self.histories['cnn'] = history
        
        return cnn, history
    
    def train_gru(self, data):
        """Entraîne le modèle GRU"""
        logger.info("🏋️  Entraînement GRU...")
        
        gru = GRUModel(input_shape=(data['X_train'].shape[1], 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'storage/models/gru_best.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Entraînement
        history = gru.model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_test'], data['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder
        gru.save_model(f"storage/models/gru_{datetime.now().strftime('%Y%m%d')}.h5")
        
        self.models['gru'] = gru
        self.histories['gru'] = history
        
        return gru, history
    
    def generate_training_report(self):
        """Génère un rapport d'entraînement"""
        report = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'performances': {}
        }
        
        for name, history in self.histories.items():
            report['performances'][name] = {
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'best_val_loss': min(history.history['val_loss']),
                'epochs_trained': len(history.history['loss'])
            }
        
        # Sauvegarder le rapport
        report_file = f"storage/models/training_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Rapport généré: {report_file}")
        
        # Générer des graphiques
        self.plot_training_history()
        
        return report
    
    def plot_training_history(self):
        """Génère des graphiques d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for idx, (name, history) in enumerate(self.histories.items()):
            ax = axes[idx]
            ax.plot(history.history['loss'], label='Train', linewidth=2)
            ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
            ax.set_title(f'Évolution de la perte - {name.upper()}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f"storage/models/training_plot_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_file, dpi=100)
        plt.close()
        
        logger.info(f"📈 Graphique sauvegardé: {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'gru', 'both'], default='both')
    parser.add_argument('--save-best', action='store_true')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(model_type=args.model)
    
    try:
        # Charger les données
        data = trainer.load_data()
        
        # Entraîner les modèles
        if args.model in ['cnn', 'both']:
            trainer.train_cnn(data)
        
        if args.model in ['gru', 'both']:
            trainer.train_gru(data)
        
        # Générer le rapport
        report = trainer.generate_training_report()
        
        print("✅ Entraînement terminé avec succès!")
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        raise