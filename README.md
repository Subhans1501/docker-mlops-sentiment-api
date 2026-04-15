# NLP Sentiment Inference Service & Docker Deployment

This repository contains a complete Machine Learning Operations (MLOps) pipeline for training, containerizing, and deploying a Natural Language Processing (NLP) model. 

The project demonstrates the ability to transition a machine learning model from a local training environment into a production-ready, containerized API using Docker and Flask.

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-learn (Naive Bayes, CountVectorizer)
* **API Framework:** Flask
* **Containerization:** Docker, Docker Hub
* **Language:** Python 3.9

## 📁 Project Structure
```text
docker-mlops-sentiment-api/
├── model/                  # Generated after training (Not tracked in Git)
│   ├── model.pkl           # Trained Naive Bayes model
│   └── vectorizer.pkl      # Trained CountVectorizer
├── app.py                  # Flask application serving the API
├── train_model.py          # Script to train and serialize the NLP model
├── Dockerfile              # Containerization instructions
├── requirements.txt        # Python dependencies
└── executed_commands.md    # Documentation of MLOps commands