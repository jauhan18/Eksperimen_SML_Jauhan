import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import mlflow
import dagshub
import os

# Inisialisasi DagsHub (Ganti dengan Username dan Nama Repo DagsHub-mu!)
dagshub.init(repo_owner='jauhan18', repo_name='Submission-MLOps', mlflow=True)

# Matikan autolog untuk manual logging
mlflow.sklearn.autolog(disable=True)

# Load Data
df = pd.read_csv('C:\\Users\\Jauhanunu\\Documents\\Pijak\\SMSML_Jauhan-Ahmad\\Eksperimen_SML_Jauhan\\Membangun_model\\dataset_preprocessing\\data_bersih.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')

with mlflow.start_run(run_name="Tuned_RandomForest"):
    # Fit Model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Prediksi
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # --- MANUAL LOGGING ---
    # 1. Log Parameter (Hasil Tuning)
    mlflow.log_params(grid_search.best_params_)
    
    # 2. Log Metrics
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
    
    # 3. Log Model
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # --- ARTEFAK TAMBAHAN (Minimal 2) ---
    # Artefak 1: Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artefak 2: Feature Importance Plot
    importances = best_model.feature_importances_
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=X.columns)
    plt.title("Feature Importances")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    
    print("Model tuning berhasil! Parameter, metrik, dan artefak telah dicatat di DagsHub.")