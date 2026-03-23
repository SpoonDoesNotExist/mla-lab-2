#!/bin/bash

# Создаем директории
mkdir -p ./logs
mkdir -p ./data

echo "========================================="
echo "Starting ML Pipeline"
echo "========================================="
echo ""

echo "Step 1: Data Creation"
echo "-----------------------------------------"
/c/Users/peple/AppData/Local/Programs/Python/Python312/python.exe data_creation.py --save_dir ./data --train_ratio 0.8
if [ $? -ne 0 ]; then
    echo "ERROR: Data creation failed"
    exit 1
fi
echo ""

echo "Step 2: Data Preprocessing"
echo "-----------------------------------------"
/c/Users/peple/AppData/Local/Programs/Python/Python312/python.exe model_preprocessing.py --data_dir ./data --standard_scaler_path ./data/scaler.pkl
if [ $? -ne 0 ]; then
    echo "ERROR: Data preprocessing failed"
    exit 1
fi
echo ""

echo "Step 3: Model Training"
echo "-----------------------------------------"
/c/Users/peple/AppData/Local/Programs/Python/Python312/python.exe model_preparation.py --train_data_path ./data/train --model_save_path ./data/model.pkl
if [ $? -ne 0 ]; then
    echo "ERROR: Model training failed"
    exit 1
fi
echo ""

echo "Step 4: Model Testing"
echo "-----------------------------------------"
/c/Users/peple/AppData/Local/Programs/Python/Python312/python.exe model_testing.py --test_data_path ./data/test --model_save_path ./data/model.pkl --scaler_path ./data/scaler.pkl
if [ $? -ne 0 ]; then
    echo "ERROR: Model testing failed"
    exit 1
fi
echo ""

echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="