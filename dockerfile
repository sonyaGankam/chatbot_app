# Utilisez une image Python officielle
FROM python:3.9-slim

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les fichiers nécessaires
COPY . .

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]