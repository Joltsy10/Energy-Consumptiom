# Energy Consumption Forecasting with LSTM

An end-to-end machine learning project for predicting household energy consumption using LSTM neural networks in PyTorch.

## Project Overview

This project uses historical energy consumption data to predict future usage patterns. Built with PyTorch, it demonstrates the full ML pipeline from data preprocessing to model training.

## Features

- Time series forecasting using LSTM
- Data preprocessing pipeline
- Model training and evaluation
- GPU acceleration support (CUDA)

## Dataset

UCI Household Power Consumption Dataset
- 2M+ measurements over 4 years
- 1-minute sampling rate
- Predicting global active power consumption

## Project Structure
```
Energy-Consumptiom/
├── api/             # Code for api    
├── data/            # Raw data files
├── models/          # Saved model weights
└── src/             # Source code modules
    ├── model.py              # LSTM architecture
    ├── data_preprocessing.py # Data pipeline
    ├── evaluation.py         # Eval of model based on baseline
    └── train.py              # Training logic
├──evaluation_pipeline.py   # Single entry for evaluation of model
└──train_pipeline.py        # Single entry for training
```

## Installation
```bash
git clone https://github.com/Joltsy10/Energy-Consumptiom.git
cd Energy-Consumptiom

pip install -r requirements.txt
```

## Usage
```python
python train_pipeline.py
```

## Model Architecture

- LSTM with 64 hidden units
- Single layer
- Input: 168-hour sequence
- Output: Next hour prediction


## Model Performance (v1.0)

**Current Status:** Baseline model using univariate time series (power consumption only)

**Metrics:**
- R² Score: 0.9442 (captures 94% of variance)
- Peak Detection: 89.60% recall
- Directional Accuracy: 44.19%
- MAE: 0.076 kW

**Baseline Comparison:**
- Naive persistence: 0.072 kW MAE
- LSTM (v1.0): 0.076 kW MAE

**Analysis:**
The naive baseline performs well due to high autocorrelation in the data (median hour-to-hour change of 6.8W). The univariate LSTM shows strong pattern recognition (94% R²) but requires temporal features to outperform persistence models.

**Planned v2 Improvements:**
- [ ] Add temporal features (hour, day of week, seasonality)
- [ ] Multi-horizon forecasting (6hr, 12hr, 24hr ahead)
- [ ] Enhanced baselines (moving average, SARIMA)

## Technologies

- Python 3.10
- PyTorch (CUDA enabled)
- Pandas
- NumPy
- Scikit-learn

## Next Steps

- [X] Deploy model as REST API
- [X] Add monitoring and logging
- [x] Dockerize the application

## Author

- Rayyan Farooq

Built as part of ML engineering portfolio
```
