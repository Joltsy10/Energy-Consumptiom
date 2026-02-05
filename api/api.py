from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import pickle
import sys
import numpy as np

sys.path.append('../src')
from src.model import LSTMModel

app = FastAPI(title= "Energy Consumption Forecasting API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel(input_size = 1, hidden_size = 64, num_layers = 1, output_size = 1)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Model and Scaler loaded successfully!")

class PredictionInput(BaseModel):
    
    values : list[float] = Field(
        ...,
        min_length = 168,
        max_length = 168,
        description= "168 hours of energy consumption values (in kW)"
    )

class PredictionOutput(BaseModel):

    prediction : float
    prediction_kw : float
    input_hours : int
    model_version : str

@app.get("/")
def root():
    return{
        "message" : "Energy Consumption Forecasting API",
        "status" : "Online",
        "endpoints" : {
            "predict" : "/predict",
            "docs" : "/docs"
        }
    }

@app.post("/predict", response_model= PredictionOutput)
def predict(input_data : PredictionInput):

    try :
        values = np.array(input_data.values).reshape(-1, 1)
        
        # Scale the input (same as training)
        scaled_values = scaler.transform(values)
        
        # Reshape for LSTM: (1, 24, 1) = (batch_size, sequence_length, features)
        X = torch.FloatTensor(scaled_values).reshape(1, 168, 1).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(X).cpu().numpy()
        
        # Inverse transform to get original scale
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        return PredictionOutput(
            prediction=float(prediction),
            prediction_kw=float(prediction),
            input_hours=24,
            model_version="v1.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Prediction failed : {str(e)}")
    
app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "device": str(device)
    }

