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
├── data/              # Raw data files
├── models/            # Saved model weights
├── notebooks/         # Jupyter notebooks for training
└── src/              # Source code modules
    ├── model.py              # LSTM architecture
    ├── data_preprocessing.py # Data pipeline
    └── train.py              # Training logic
```

## Installation
```bash
git clone https://github.com/Joltsy10/Energy-Consumptiom.git
cd Energy-Consumptiom

pip install -r requirements.txt
```

## Usage
```python
jupyter notebook notebooks/train_model.ipynb
```

## Model Architecture

- LSTM with 50 hidden units
- Single layer
- Input: 24-hour sequence
- Output: Next hour prediction

## Results

- Training Loss: 0.000509
- Validation Loss: 0.000454
- Test Loss: 0.000352

## Model Performance

- **Accuracy**: 94.4% R² score on test set
- **Error**: Mean absolute error of 0.08 kW (~8% MAPE)
- **Peak Detection**: 90% recall, 88% precision
- **Robustness**: Consistent performance across train/val/test splits

## Technologies

- Python 3.10
- PyTorch (CUDA enabled)
- Pandas
- NumPy
- Scikit-learn

## Next Steps

- [ ] Deploy model as REST API
- [ ] Add monitoring and logging
- [ ] Dockerize the application
- [ ] Implement CI/CD pipeline

## Author

- Rayyan Farooq

Built as part of ML engineering portfolio
```
