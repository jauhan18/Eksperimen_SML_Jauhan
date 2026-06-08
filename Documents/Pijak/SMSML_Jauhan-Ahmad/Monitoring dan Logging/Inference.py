import time
import os
import psutil  # Mengambil data resource asli komputer
import pandas as pd
import mlflow.sklearn
from flask import Flask, request, jsonify

# IMPORT 10 METRIK & FUNGSI DARI FILE EXPORTER
from prometheus_exporter import (
    REQUEST_COUNT, ERROR_COUNT, LATENCY, ACTIVE_REQUESTS,
    PRED_SURVIVED, PRED_NOT_SURVIVED, MODEL_CONFIDENCE, DATA_DRIFT,
    CPU_USAGE, MEMORY_USAGE, start_prometheus_server
)

app = Flask(__name__)

# Jalankan server Prometheus port 8000 menggunakan fungsi dari exporter
start_prometheus_server(8000)

# Load Model Titanic Aktual menggunakan skema file:/// agar lolos eror Windows
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model_artifact")

# Mengubah backslash Windows (\) menjadi forward slash (/) dan menambah skema file:///
MODEL_URI = f"file:///{MODEL_DIR.replace(os.sep, '/')}"

model = mlflow.sklearn.load_model(MODEL_URI)

@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    start_time = time.time()  # Catat awal waktu latency
    
    # Ambil data hardware asli laptop lewat psutil
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().used / (1024 * 1024))
    
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        
        # 1. Prediksi Model Aktual
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        
        # 2. Ambil Confidence Score Asli Model
        confidence = float(max(probabilities))
        MODEL_CONFIDENCE.set(confidence)
        
        # 3. Catat Kelas Output Asli Model
        if prediction == 1:
            PRED_SURVIVED.inc()
        else:
            PRED_NOT_SURVIVED.inc()
            
        # 4. Hitung Data Drift Aktual (Deviasi nilai Age input terhadap baseline ~29.6)
        if 'Age' in df_input.columns and not df_input['Age'].isnull().all():
            baseline_age = 29.6
            drift_score = abs(df_input['Age'].mean() - baseline_age) / baseline_age
            DATA_DRIFT.set(min(drift_score, 1.0))
        else:
            DATA_DRIFT.set(0.02)
            
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'confidence': confidence
        })
        
    except Exception as e:
        ERROR_COUNT.inc()  # Hanya bertambah jika sistem benar-benar crash
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
    finally:
        LATENCY.observe(time.time() - start_time)  # Catat durasi real
        ACTIVE_REQUESTS.dec()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)