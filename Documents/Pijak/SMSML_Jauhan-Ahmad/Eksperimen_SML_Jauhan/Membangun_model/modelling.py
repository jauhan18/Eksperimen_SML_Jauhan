import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import dagshub

# Inisialisasi DagsHub (Ganti dengan Username dan Nama Repo DagsHub-mu!)
dagshub.init(repo_owner='jauhan18', repo_name='Submission-MLOps', mlflow=True)

# 1. Load Data
df = pd.read_csv('C:\\Users\\Jauhanunu\\Documents\\Pijak\\SMSML_Jauhan-Ahmad\\Eksperimen_SML_Jauhan\\Membangun_model\\dataset_preprocessing\\data_bersih.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Autolog MLflow
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model berhasil dilatih dengan autolog!")