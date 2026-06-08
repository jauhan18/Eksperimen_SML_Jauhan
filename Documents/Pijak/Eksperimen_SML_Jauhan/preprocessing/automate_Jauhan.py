import pandas as pd
import os

def load_data(filepath):
    print("Loading data...")
    return pd.read_csv(filepath)

def clean_text(text):
    # Masukkan logika pembersihan data di sini
    # Contoh: mengubah teks menjadi lowercase
    return str(text).lower()

def preprocess_data(df):
    print("Preprocessing data...")
    # Sesuaikan 'review' dengan nama kolom di datasetmu
    if 'review' in df.columns:
        df['cleaned_review'] = df['review'].apply(clean_text)
    return df

def save_data(df, output_path):
    print("Saving preprocessed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data tersimpan di {output_path}")

if __name__ == "__main__":
    # Jalur relatif menyesuaikan posisi script saat dijalankan
    input_file = "../dataset_raw/data_mentah.csv"
    output_file = "dataset_preprocessing/data_bersih.csv"

    raw_df = load_data(input_file)
    clean_df = preprocess_data(raw_df)
    save_data(clean_df, output_file)