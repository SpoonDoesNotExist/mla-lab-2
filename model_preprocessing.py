import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

os.makedirs('logs', exist_ok=True)


logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def preprocess(data_dir, scaler_save_path):
    """Предобрабатывает данные и сохраняет scaler"""
    try:
        logging.info("=" * 50)
        logging.info("START: Data Preprocessing")
        logging.info("=" * 50)
        
        data_dir = Path(data_dir)
        train_dir = data_dir / "train"
        
        if not train_dir.exists():
            raise Exception(f"Train directory {train_dir} does not exist")
        
        # Собираем все тренировочные данные
        all_temperatures = []
        files_loaded = []
        
        print("Загрузка тренировочных данных...")
        logging.info("Loading training data...")
        
        for filename in os.listdir(train_dir):
            if filename.endswith(".csv"):
                filepath = train_dir / filename
                df = pd.read_csv(filepath)
                all_temperatures.extend(df['temperature'].values)
                files_loaded.append(filename)
                print(f"  Загружен {filename}: {len(df)} записей")
                logging.info(f"Loaded {filename}: {len(df)} rows")
        
        # Обучаем StandardScaler
        X_temperatures = np.array(all_temperatures).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(X_temperatures)
        
        print(f"\nStandardScaler обучен:")
        print(f"  Среднее (mean): {scaler.mean_[0]:.2f}")
        print(f"  Стандартное отклонение (scale): {scaler.scale_[0]:.2f}")
        
        logging.info(f"Scaler trained: mean={scaler.mean_[0]:.2f}, scale={scaler.scale_[0]:.2f}")
        
        # Применяем масштабирование
        print("\nПрименение масштабирования...")
        logging.info("Applying scaling...")
        
        for filename in os.listdir(train_dir):
            if filename.endswith(".csv"):
                filepath = train_dir / filename
                df = pd.read_csv(filepath)
                df['temperature_scaled'] = scaler.transform(df[['temperature']]).flatten()
                df.to_csv(filepath, index=False)
                print(f"  Обработан {filename}")
                logging.info(f"Scaled {filename}")
        
        # Сохраняем scaler
        Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)
        
        print(f"\n✅ Предобработка завершена!")
        print(f"  Scaler сохранен в {scaler_save_path}")
        
        logging.info(f"Scaler saved to {scaler_save_path}")
        logging.info(f"Processed {len(files_loaded)} training files")
        logging.info("=" * 50)
        
        return scaler
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        print(f"Ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--data_dir", 
                        help="Path to train/test directories", 
                        default="./data")
    parser.add_argument("--standard_scaler_path", 
                        help="Path to save StandardScaler", 
                        default="./data/scaler.pkl")
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.standard_scaler_path)