import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(predictions, actual, scaler = None):

    if scaler is not None:
        predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_orig = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    else:
        predictions_orig = predictions
        actual_orig = actual

    metrics = {}

    metrics['mae'] = mean_absolute_error(actual_orig, predictions_orig)
    metrics['mse'] = mean_squared_error(actual_orig, predictions_orig)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(actual_orig, predictions_orig)

    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((actual_orig - predictions_orig) / actual_orig)
        mape = mape[np.isfinite(mape)]  # Remove inf/nan
        metrics['mape'] = np.mean(mape) * 100

    if len(actual_orig) > 1:
        actual_direction = np.sign(actual_orig[1:] - actual_orig[:-1])
        pred_direction = np.sign(predictions_orig[1:] - predictions_orig[:-1])
        metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction) * 100

    threshold = np.percentile(actual_orig, 90)
    actual_peaks = actual_orig > threshold
    pred_peaks = predictions_orig > threshold
    
    if np.sum(pred_peaks) > 0:
        metrics['peak_precision'] = np.sum(actual_peaks & pred_peaks) / np.sum(pred_peaks) * 100
    else:
        metrics['peak_precision'] = 0
    
    if np.sum(actual_peaks) > 0:
        metrics['peak_recall'] = np.sum(actual_peaks & pred_peaks) / np.sum(actual_peaks) * 100
    else:
        metrics['peak_recall'] = 0
    
    return metrics

def print_metrics(metrics):
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"MAE (Mean Absolute Error):      {metrics['mae']:.4f} kW")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f} kW")
    print(f"MAPE (Mean Abs Percentage Err): {metrics['mape']:.2f}%")
    print(f"R² Score:                       {metrics['r2']:.4f}")
    print(f"\nDirectional Accuracy:           {metrics['directional_accuracy']:.2f}%")
    print(f"Peak Detection Precision:       {metrics['peak_precision']:.2f}%")
    print(f"Peak Detection Recall:          {metrics['peak_recall']:.2f}%")
    print("="*50 + "\n")