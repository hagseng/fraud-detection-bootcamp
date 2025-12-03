import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing import preprocess_data

def load_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {config.MODEL_PATH}")
    
    model = joblib.load(config.MODEL_PATH)
    print(f"Model başarıyla yüklendi: {config.MODEL_PATH}")
    return model

def make_prediction(df: pd.DataFrame):
    model = load_model()
    df_processed = preprocess_data(df)

    features = [col for col in df_processed.columns if col not in ['Class', 'Time', 'Event_Block']]
    X = df_processed[features]

    scores = model.predict(X)
    return scores

if __name__ == "__main__":
    try:
        sample_df = pd.read_csv(config.RAW_DATA_PATH).sample(5)
        print("Rastgele 5 satır seçildi.")
        
        predictions = make_prediction(sample_df)
        
        print("\n--- TAHMİN SONUÇLARI (Risk Skorları) ---")
        print(predictions)
        print("------------------------------------------")
        print("Sistem başarıyla çalışıyor!")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
