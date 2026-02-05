import torch
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error

from src.data_preprocessing import load_data, prepare_data
from src.model import LSTMModel
from src.evaluation import evaluate_model, print_metrics

# ============================================================================
#                               CONFIGURATION
# ============================================================================
DATA_PATH = 'data/household_power_consumption.txt'
MODEL_PATH = 'models/best_model.pth'
SCALER_PATH = 'models/scaler.pkl'

SEQ_LEN = 168
HIDDEN_SIZE = 64
NUM_LAYERS = 1

# ============================================================================
#                               BASELINE MODELS
# ============================================================================

def naive_baseline(y_test):
    predictions = y_test[:-1]
    actual = y_test[1:]
    
    return predictions, actual

def seasonal_naive_baseline(y_test, season = 24):
    predictions = y_test[:-season]
    actual = y_test[season:]

    return predictions, actual

def moving_average_baseline(y_test, window = 24):
    predictions = []
    actuals = []

    for i in range(window, len(y_test)):
        pred = np.mean(y_test[i-window:i])
        predictions.append(pred)
        actuals.append(y_test[i])

    return np.array(predictions), np.array(actuals)

# ============================================================================
#                           MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    print("="*60)
    print("ENERGY CONSUMPTION FORECASTING - EVALUATION PIPELINE")
    print("="*60)

    df = load_data(DATA_PATH)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler = prepare_data(
        df, seq_length=SEQ_LEN
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(
        input_size= 1,
        hidden_size= HIDDEN_SIZE,
        num_layers= NUM_LAYERS,
        output_size= 1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    def get_predictions_batched(model, data, device, batch_size=1024):
        """Get predictions in batches to avoid OOM errors."""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i+batch_size].to(device)
                batch_pred = model(batch).cpu().numpy()
                predictions.append(batch_pred)
        
        return np.concatenate(predictions).flatten()
    
    lstm_predictions = get_predictions_batched(model, x_test, device, batch_size=1024)

    print(f"\n" + "="*60)
    print("LSTM MODEL PERFORMANCE")
    print("="*60)
    lstm_metrics = evaluate_model(lstm_predictions, y_test, scaler)
    print_metrics(lstm_metrics)

    print("\n1. NAIVE BASELINE (predict last value)")
    print("-"*60)
    naive_pred, naive_actual = naive_baseline(y_test)
    naive_metrics = evaluate_model(naive_pred, naive_actual, scaler)
    print(f"MAE:  {naive_metrics['mae']:.4f} kW")
    print(f"RMSE: {naive_metrics['rmse']:.4f} kW")
    print(f"MAPE: {naive_metrics['mape']:.2f}%")
    
    # Seasonal naive
    print("\n2. SEASONAL NAIVE BASELINE (predict 24h ago)")
    print("-"*60)
    seasonal_pred, seasonal_actual = seasonal_naive_baseline(y_test, season=24)
    seasonal_metrics = evaluate_model(seasonal_pred, seasonal_actual, scaler)
    print(f"MAE:  {seasonal_metrics['mae']:.4f} kW")
    print(f"RMSE: {seasonal_metrics['rmse']:.4f} kW")
    print(f"MAPE: {seasonal_metrics['mape']:.2f}%")
    
    # Moving average
    print("\n3. MOVING AVERAGE BASELINE (24h window)")
    print("-"*60)
    ma_pred, ma_actual = moving_average_baseline(y_test, window=24)
    ma_metrics = evaluate_model(ma_pred, ma_actual, scaler)
    print(f"MAE:  {ma_metrics['mae']:.4f} kW")
    print(f"RMSE: {ma_metrics['rmse']:.4f} kW")
    print(f"MAPE: {ma_metrics['mape']:.2f}%")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Model':<25} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'MAPE (%)':<10}")
    print("-"*60)
    print(f"{'Naive (t-1)':<25} | {naive_metrics['mae']:<10.4f} | {naive_metrics['rmse']:<10.4f} | {naive_metrics['mape']:<10.2f}")
    print(f"{'Seasonal Naive (t-24)':<25} | {seasonal_metrics['mae']:<10.4f} | {seasonal_metrics['rmse']:<10.4f} | {seasonal_metrics['mape']:<10.2f}")
    print(f"{'Moving Average (24h)':<25} | {ma_metrics['mae']:<10.4f} | {ma_metrics['rmse']:<10.4f} | {ma_metrics['mape']:<10.2f}")
    print(f"{'LSTM (V1)':<25} | {lstm_metrics['mae']:<10.4f} | {lstm_metrics['rmse']:<10.4f} | {lstm_metrics['mape']:<10.2f}")
    print("-"*60)

if __name__ == '__main__':
    main()