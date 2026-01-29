#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, ticker='AAPL', data_dir='storage/data'):
        self.ticker = ticker
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_stock_data(self, period='4y'):
        """Télécharge les données depuis Yahoo Finance"""
        logger.info(f"Téléchargement des données pour {self.ticker}...")
        
        try:
            data = yf.download(self.ticker, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"Aucune donnée téléchargée pour {self.ticker}")
            
            # Ajouter des métadonnées
            metadata = {
                'ticker': self.ticker,
                'download_date': datetime.now().isoformat(),
                'period': period,
                'rows': len(data),
                'columns': list(data.columns)
            }
            
            # Sauvegarder les données
            filename = f"{self.ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            data.to_csv(filepath)
            
            # Sauvegarder les métadonnées
            metafile = filepath.replace('.csv', '_meta.json')
            with open(metafile, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Données sauvegardées: {filepath} ({len(data)} lignes)")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du téléchargement: {e}")
            raise
    
    def download_recent_data(self, days=7):
        """Télécharge les données récentes pour mise à jour"""
        logger.info(f"Mise à jour des données pour {self.ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(
                self.ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour: {e}")
            return None

if __name__ == "__main__":
    collector = DataCollector()
    data = collector.download_stock_data()
    print(f"Données collectées: {data.shape}")