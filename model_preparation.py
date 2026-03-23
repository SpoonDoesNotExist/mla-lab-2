import argparse
import logging
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')


os.makedirs('logs', exist_ok=True)


logging.basicConfig(
    filename='logs/model_prepatation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_training_data(train_data_path):
    """Загружает тренировочные данные из указанного пути"""
    logging.info(f"Loading training data from {train_data_path}")
    
    # Если передан файл
    if Path(train_data_path).is_file():
        df = pd.read_csv(train_data_path)
        X = df['day'].values.reshape(-1, 1)
        
        # Проверяем наличие масштабированной колонки
        if 'temperature_scaled' in df.columns:
            y = df['temperature_scaled'].values
        else:
            y = df['temperature'].values
            
        logging.info(f"Loaded single file: {len(df)} samples")
        return X, y
    
    # Если передана директория
    elif Path(train_data_path).is_dir():
        X_train = []
        y_train = []
        
        for filename in os.listdir(train_data_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(train_data_path, filename)
                df = pd.read_csv(filepath)
                X_train.extend(df['day'].values)
                
                if 'temperature_scaled' in df.columns:
                    y_train.extend(df['temperature_scaled'].values)
                else:
                    y_train.extend(df['temperature'].values)
                    
                logging.info(f"Loaded {filename}: {len(df)} samples")
        
        X = np.array(X_train).reshape(-1, 1)
        y = np.array(y_train)
        logging.info(f"Total: {len(X)} samples from {train_data_path}")
        return X, y
    
    else:
        raise Exception(f"Path {train_data_path} does not exist")

def train(train_data_path, model_save_path):
    """Обучает модель и сохраняет её"""
    try:
        logging.info("=" * 50)
        logging.info("START: Model Training")
        logging.info("=" * 50)
        
        # Загружаем данные
        X_train, y_train = load_training_data(train_data_path)
        
        print(f"\nДанные для обучения:")
        print(f"  Количество образцов: {len(X_train)}")
        print(f"  Диапазон дней: от {X_train.min()} до {X_train.max()}")
        print(f"  Диапазон температур: от {y_train.min():.2f} до {y_train.max():.2f}")
        
        logging.info(f"Training data: {len(X_train)} samples")
        logging.info(f"Day range: [{X_train.min()}, {X_train.max()}]")
        logging.info(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        # Обучаем модель
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Оцениваем модель
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        print(f"\nРезультаты обучения:")
        print(f"  Коэффициент (slope): {model.coef_[0]:.4f}")
        print(f"  Смещение (intercept): {model.intercept_:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 score: {r2:.4f}")
        
        logging.info(f"Training results:")
        logging.info(f"  Coefficient: {model.coef_[0]:.4f}")
        logging.info(f"  Intercept: {model.intercept_:.4f}")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  R2: {r2:.4f}")
        
        # Сохраняем модель
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"\n✅ Модель сохранена в {model_save_path}")
        logging.info(f"Model saved to {model_save_path}")
        logging.info("=" * 50)
        
        return model
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        print(f"Ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument("--train_data_path", 
                        help="Path to preprocessed train data (file or directory)", 
                        required=True)
    parser.add_argument("--model_save_path", 
                        help="Path to save model weights (.pkl)", 
                        required=True)
    args = parser.parse_args()
    
    train(args.train_data_path, args.model_save_path)