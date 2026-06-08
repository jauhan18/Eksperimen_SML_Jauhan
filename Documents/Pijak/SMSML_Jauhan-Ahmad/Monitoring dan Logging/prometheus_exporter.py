from prometheus_client import start_http_server, Counter, Histogram, Gauge

# =====================================================================
# DEFINISI 10 METRIK TARGET ADVANCED (100% REAL DATA, NO DUMMY)
# =====================================================================
REQUEST_COUNT = Counter('app_request_count', 'Total request inference yang benar-benar masuk')
ERROR_COUNT = Counter('app_error_count', 'Error prediction yang benar-benar terjadi saat inference gagal')
LATENCY = Histogram('app_request_latency_seconds', 'Latency inference aktual saat model melakukan prediksi')
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Jumlah request yang sedang aktif diproses')

# Hasil Output Model Aktual
PRED_SURVIVED = Counter('app_pred_survived_count', 'Jumlah kelas Selamat (1) dari output model')
PRED_NOT_SURVIVED = Counter('app_pred_not_survived_count', 'Jumlah kelas Tidak Selamat (0) dari output model')
MODEL_CONFIDENCE = Gauge('app_model_confidence_score', 'Confidence score hasil prediksi aktual model')
DATA_DRIFT = Gauge('app_data_drift_score', 'Monitoring data drift berdasarkan deviasi data input')

# Resource Hardware Server Aktual
CPU_USAGE = Gauge('app_cpu_usage_percent', 'Monitoring resource aktual CPU server menggunakan psutil')
MEMORY_USAGE = Gauge('app_memory_usage_mb', 'Monitoring resource aktual Memory server dalam MB')

def start_prometheus_server(port=8000):
    """Fungsi untuk menyalakan server metrics Prometheus"""
    start_http_server(port)
    print(f"Prometheus metrics server siap berjalan di port {port}...")