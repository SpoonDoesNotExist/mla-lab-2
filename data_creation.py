import argparse
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import os
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/data_creation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def generate_temperature_data(days=30, base_temp=20, seasonal_amplitude=8,
                              noise_level=2.0, anomaly_prob=0.1, anomaly_magnitude=10):
    """Генерирует синтетические данные температуры"""
    days_array = np.arange(days)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * days_array / 365)
    trend = 0.05 * days_array
    noise = np.random.normal(0, noise_level, days)
    temperature = base_temp + seasonal + trend + noise

    # Добавляем аномалии
    for i in range(days):
        if random.random() < anomaly_prob:
            anomaly = random.choice([-1, 1]) * anomaly_magnitude * random.uniform(0.5, 1.5)
            temperature[i] += anomaly

    df = pd.DataFrame({
        'day': days_array,
        'temperature': temperature,
        'is_anomaly': [random.random() < anomaly_prob for _ in range(days)]
    })
    return df

def create_data(save_dir, train_ratio=0.8):
    """Создает наборы данных и сохраняет их"""
    save_path = Path(save_dir)
    train_dir = save_path / "train"
    test_dir = save_path / "test"
    
    # Создаем директории
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 50)
    logging.info("START: Data Creation")
    logging.info("=" * 50)
    
    # Конфигурации для тренировочных данных
    train_configs = [
        {'days': 30, 'base_temp': 15, 'noise_level': 1.5, 'anomaly_prob': 0.05},
        {'days': 35, 'base_temp': 17, 'noise_level': 2.0, 'anomaly_prob': 0.07},
        {'days': 40, 'base_temp': 19, 'noise_level': 2.5, 'anomaly_prob': 0.09},
        {'days': 45, 'base_temp': 21, 'noise_level': 3.0, 'anomaly_prob': 0.11},
        {'days': 50, 'base_temp': 23, 'noise_level': 3.5, 'anomaly_prob': 0.13}
    ]
    
    # Конфигурации для тестовых данных
    test_configs = [
        {'days': 30, 'base_temp': 20, 'noise_level': 2.0, 'anomaly_prob': 0.08},
        {'days': 33, 'base_temp': 20, 'noise_level': 2.0, 'anomaly_prob': 0.08},
        {'days': 36, 'base_temp': 20, 'noise_level': 2.0, 'anomaly_prob': 0.08}
    ]
    
    print("Генерация тренировочных данных...")
    logging.info("Generating training data...")
    
    for i, config in enumerate(train_configs):
        df = generate_temperature_data(**config)
        filepath = train_dir / f"data_{i}.csv"
        df.to_csv(filepath, index=False)
        print(f"  Создан train/data_{i}.csv с {len(df)} записями")
        logging.info(f"Created {filepath}: {len(df)} rows")
    
    print("\nГенерация тестовых данных...")
    logging.info("Generating test data...")
    
    for i, config in enumerate(test_configs):
        df = generate_temperature_data(**config)
        filepath = test_dir / f"data_{i}.csv"
        df.to_csv(filepath, index=False)
        print(f"  Создан test/data_{i}.csv с {len(df)} записями")
        logging.info(f"Created {filepath}: {len(df)} rows")
    
    train_count = len(list(train_dir.glob("*.csv")))
    test_count = len(list(test_dir.glob("*.csv")))
    
    print(f"\n✅ Данные успешно созданы!")
    print(f"  - train: {train_count} файлов в {train_dir}")
    print(f"  - test: {test_count} файлов в {test_dir}")
    
    logging.info(f"Data creation completed: train={train_count} files, test={test_count} files")
    logging.info("=" * 50)
    
    return train_dir, test_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data creation script")
    parser.add_argument("--save_dir", 
                        help="Path to save data", 
                        default="./data")
    parser.add_argument("--train_ratio", 
                        help="Train/test split ratio", 
                        type=float, 
                        default=0.8)
    args = parser.parse_args()
    
    create_data(args.save_dir, args.train_ratio)