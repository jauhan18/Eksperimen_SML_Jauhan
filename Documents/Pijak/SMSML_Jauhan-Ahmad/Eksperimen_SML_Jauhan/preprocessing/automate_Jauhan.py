import pandas as pd
import os

def load_data(filepath):
    print("Loading data...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("Preprocessing data...")
    df_clean = df.copy()
    
    # Drop kolom tidak relevan
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], inplace=True)
    
    # Imputasi Missing Values
    if 'Age' in df_clean.columns:
        df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
        
    # Encoding Data Kategorikal
    if 'Sex' in df_clean.columns:
        df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
        
    if 'Embarked' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)
        
    return df_clean

def save_data(df, output_path):
    print("Saving preprocessed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data tersimpan di {output_path}")

if __name__ == "__main__":
    input_file = "../dataset_raw/Titanic-Dataset.csv"
    output_file = "dataset_preprocessing/data_bersih.csv"

    raw_df = load_data(input_file)
    clean_df = preprocess_data(raw_df)
    save_data(clean_df, output_file)