import mlflow
import mlflow.sklearn
import joblib
import sqlite3
from datetime import datetime

# Setup MLflow with local SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flipkart_production")

# Log your Stage 1 model to MLflow
with mlflow.start_run(run_name="logistic_production"):
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("f1_score", 0.86)
    mlflow.log_param("features", 5000)
    mlflow.log_param("dataset_size", 100000)
    
print("✅ Model logged to MLflow!")
print("📊 View experiments: mlflow ui")

