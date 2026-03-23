import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_model_and_scaler(model_path, scaler_path=None):
    """Загружает модель и scaler"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {model_path}")
    
    # Если scaler_path не указан, ищем в той же директории
    if scaler_path is None:
        scaler_path = Path(model_path).parent / "scaler.pkl"
    
    if Path(scaler_path).exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logging.info(f"Scaler loaded from {scaler_path}")
    else:
        scaler = None
        logging.warning("Scaler not found, using original values")
    
    return model, scaler

def test(test_data_path, model_save_path, scaler_path=None):
    """Тестирует модель на тестовых данных"""
    try:
        logging.info("=" * 50)
        logging.info("START: Model Testing")
        logging.info("=" * 50)
        
        # Загружаем модель и scaler
        model, scaler = load_model_and_scaler(model_save_path, scaler_path)
        print("Модель загружена")
        
        # Загружаем тестовые данные
        if Path(test_data_path).is_file():
            test_files = [test_data_path]
        else:
            test_files = [os.path.join(test_data_path, f) for f in os.listdir(test_data_path) if f.endswith(".csv")]
        
        logging.info(f"Found {len(test_files)} test files")
        
        all_results = []
        print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        
        for filepath in test_files:
            df = pd.read_csv(filepath)
            filename = os.path.basename(filepath)
            
            # Подготовка данных
            X_test = df[['day']].values
            
            if scaler and 'temperature' in df.columns:
                y_test_scaled = scaler.transform(df[['temperature']]).flatten()
                y_pred_scaled = model.predict(X_test)
                
                # Обратное масштабирование
                y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
                y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                mae = mean_absolute_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                r2 = r2_score(y_test_scaled, y_pred_scaled)
                
            else:
                # Если нет scaler, используем исходные значения
                y_test = df['temperature'].values
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
            
            result = {
                'file': filename,
                'samples': len(df),
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            all_results.append(result)
            
            print(f"\nФайл: {filename}")
            print(f"  Количество записей: {len(df)}")
            print(f"  MAE: {mae:.2f}°C")
            print(f"  RMSE: {rmse:.2f}°C")
            print(f"  R2: {r2:.4f}")
            
            logging.info(f"Test on {filename}: samples={len(df)}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        
        # Средние результаты
        if all_results:
            print("\nСРЕДНИЕ РЕЗУЛЬТАТЫ")
            avg_mae = np.mean([r['mae'] for r in all_results])
            avg_rmse = np.mean([r['rmse'] for r in all_results])
            avg_r2 = np.mean([r['r2'] for r in all_results])
            
            print(f"  Средняя MAE: {avg_mae:.2f}°C")
            print(f"  Средняя RMSE: {avg_rmse:.2f}°C")
            print(f"  Средний R2: {avg_r2:.4f}")
            
            logging.info(f"Average results: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, R2={avg_r2:.4f}")
            
            # Сохраняем результаты
            results_df = pd.DataFrame(all_results)
            results_df.to_csv("test_results.csv", index=False)
            print("\n✅ Результаты сохранены в test_results.csv")
        
        logging.info("=" * 50)
        return all_results
        
    except Exception as e:
        logging.error(f"Error during model testing: {str(e)}")
        print(f"Ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument("--test_data_path", 
                        help="Path to preprocessed test data (file or directory)", 
                        required=True)
    parser.add_argument("--model_save_path", 
                        help="Path to saved model weights (.pkl)", 
                        required=True)
    parser.add_argument("--scaler_path", 
                        help="Path to scaler (.pkl)", 
                        required=False)
    args = parser.parse_args()
    
    test(args.test_data_path, args.model_save_path, args.scaler_path)