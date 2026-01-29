#!/usr/bin/env python
# -*- coding: utf-8 -*-

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import json
import os
from datetime import datetime
import logging
from jinja2 import Template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailSender:
    def __init__(self, config_file='config/email_config.json'):
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file):
        """Charge la configuration email"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Configuration par défaut
            return {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": os.getenv("EMAIL_SENDER"),
                "sender_password": os.getenv("EMAIL_PASSWORD"),
                "recipients": {
                    "training": ["data-team@company.com"],
                    "alerts": ["traders@company.com"],
                    "errors": ["devops@company.com"]
                }
            }
    
    def create_training_email(self, report_data):
        """Crée un email pour le rapport d'entraînement"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .header { background: #FF9500; color: white; padding: 20px; }
                .content { padding: 20px; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Rapport d'Entraînement AAPL</h1>
                <p>Date: {{ date }}</p>
            </div>
            <div class="content">
                <h2>Performances des Modèles</h2>
                
                {% for model, metrics in performances.items() %}
                <div class="metric">
                    <h3>Modèle {{ model|upper }}</h3>
                    <p>Perte finale (validation): <b>{{ "%.4f"|format(metrics.final_val_loss) }}</b></p>
                    <p>Meilleure perte: <b>{{ "%.4f"|format(metrics.best_val_loss) }}</b></p>
                    <p>Epochs: {{ metrics.epochs_trained }}</p>
                </div>
                {% endfor %}
                
                <h2>Recommandations</h2>
                <p>Modèle recommandé: <b>{{ best_model|upper }}</b></p>
                
                <hr>
                <p><i>Ce rapport a été généré automatiquement par le système de prédiction AAPL.</i></p>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        
        # Déterminer le meilleur modèle
        best_model = min(
            report_data['performances'].items(),
            key=lambda x: x[1]['best_val_loss']
        )[0]
        
        html_content = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            performances=report_data['performances'],
            best_model=best_model
        )
        
        return html_content
    
    def create_alert_email(self, prediction_data):
        """Crée un email d'alerte opportunité"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .header { background: {% if change > 0 %}#4CAF50{% else %}#F44336{% endif %}; 
                         color: white; padding: 20px; }
                .content { padding: 20px; }
                .prediction { font-size: 24px; font-weight: bold; }
                .dates { color: #666; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Opportunité AAPL Détectée</h1>
                <p>Date d'analyse: {{ date }}</p>
            </div>
            <div class="content">
                <h2>Prédictions sur 21 jours</h2>
                
                <div class="prediction">
                    {{ first_price }} → {{ last_price }}
                    <span style="color: {% if change > 0 %}green{% else %}red{% endif %};">
                        ({{ change }}%)
                    </span>
                </div>
                
                <div class="dates">
                    Du {{ start_date }} au {{ end_date }}
                </div>
                
                <h3>Détails des prédictions:</h3>
                <ul>
                {% for date, price in predictions %}
                    <li>{{ date }}: ${{ "%.2f"|format(price) }}</li>
                {% endfor %}
                </ul>
                
                <h3>Recommandation:</h3>
                <p><b>{{ recommendation }}</b></p>
                
                <p><i>Graphique des prédictions en pièce jointe.</i></p>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        
        # Créer les paires date/prix
        predictions = list(zip(
            prediction_data['prediction_dates'],
            prediction_data['predictions']
        ))
        
        # Générer la recommandation
        change = prediction_data['total_change']
        if change > 5:
            recommendation = "ACHAT FORT - Forte tendance haussière détectée"
        elif change > 2:
            recommendation = "ACHAT MODÉRÉ - Tendance haussière"
        elif change < -3:
            recommendation = "VENTE - Tendance baissière"
        else:
            recommendation = "MAINTIEN - Stabilité du marché"
        
        html_content = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            first_price=f"${prediction_data['first_price']:.2f}",
            last_price=f"${prediction_data['last_price']:.2f}",
            change=f"{prediction_data['total_change']:.2f}",
            start_date=prediction_data['prediction_dates'][0][:10],
            end_date=prediction_data['prediction_dates'][-1][:10],
            predictions=predictions[:10],  # Afficher seulement les 10 premiers
            recommendation=recommendation
        )
        
        return html_content
    
    def send_email(self, to_emails, subject, html_content, attachments=None):
        """Envoie un email"""
        msg = MIMEMultipart()
        msg['From'] = self.config['sender_email']
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject
        
        # Corps du message
        msg.attach(MIMEText(html_content, 'html'))
        
        # Pièces jointes
        if attachments:
            for attachment in attachments:
                if os.path.exists(attachment):
                    part = MIMEBase('application', 'octet-stream')
                    with open(attachment, 'rb') as file:
                        part.set_payload(file.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={os.path.basename(attachment)}'
                    )
                    msg.attach(part)
        
        # Envoi
        try:
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['sender_email'], self.config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ Email envoyé à {len(to_emails)} destinataire(s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi email: {e}")
            return False
    
    def send_training_report(self, report_file):
        """Envoie le rapport d'entraînement"""
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        html_content = self.create_training_email(report_data)
        recipients = self.config['recipients']['training']
        
        subject = f"📊 Rapport Entraînement AAPL - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Trouver les pièces jointes (graphiques)
        plot_dir = 'storage/models'
        attachments = []
        for file in os.listdir(plot_dir):
            if file.startswith('training_plot') and file.endswith('.png'):
                attachments.append(os.path.join(plot_dir, file))
        
        return self.send_email(recipients, subject, html_content, attachments)
    
    def send_market_alert(self, prediction_file):
        """Envoie une alerte opportunité marché"""
        with open(prediction_file, 'r') as f:
            prediction_data = json.load(f)
        
        html_content = self.create_alert_email(prediction_data)
        recipients = self.config['recipients']['alerts']
        
        change = prediction_data['total_change']
        if change > 0:
            emoji = "🚀"
            trend = "HAUSSE"
        else:
            emoji = "⚠️"
            trend = "BAISSE"
        
        subject = f"{emoji} Alerte AAPL {trend} {change:.2f}% - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Pièces jointes
        plot_dir = 'storage/predictions'
        attachments = []
        for file in os.listdir(plot_dir):
            if file.startswith('prediction_plot') and file.endswith('.png'):
                attachments.append(os.path.join(plot_dir, file))
        
        return self.send_email(recipients, subject, html_content, attachments)

if __name__ == "__main__":
    import sys
    
    sender = EmailSender()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'training':
            # Trouver le dernier rapport
            import glob
            report_files = glob.glob('storage/models/training_report_*.json')
            if report_files:
                latest_report = max(report_files)
                sender.send_training_report(latest_report)
            else:
                logger.error("Aucun rapport d'entraînement trouvé")
                
        elif sys.argv[1] == 'alert':
            # Trouver les dernières prédictions
            import glob
            pred_files = glob.glob('storage/predictions/predictions_*.json')
            if pred_files:
                latest_pred = max(pred_files)
                sender.send_market_alert(latest_pred)
            else:
                logger.error("Aucune prédiction trouvée")