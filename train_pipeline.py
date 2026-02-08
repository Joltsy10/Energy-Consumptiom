import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path

from src.data_preprocessing import load_data, prepare_data
from src.model import LSTMModel

# ============================================================================
#                               CONFIGURATION 
# ============================================================================

RANDOM_SEED = 42
DATA_PATH = "data/household_power_consumption.txt"
MODEL_SAVE_PATH = "models/v2.0/best_model.pth"
X_SCALER_SAVE_PATH = "models/v2.0/x_scaler.pkl"
Y_SCALER_SAVE_PATH = "models/v2.0/y_scaler.pkl"

SEQ_LENGTH = 168
BATCH_SIZE = 64
HIDDEN_SIZE = 96
NUM_LAYERS = 1
LEARNING_RATE = 0.00015
NUM_EPOCHS = 50

EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.000005

# ============================================================================
#                                  FUNCTIONS
# ============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(x_batch)
        loss = criterion(predictions.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions.squeeze(), y_batch)

            total_loss += loss

    return total_loss / len(dataloader)

# ============================================================================
#                               MAIN TRAIN PIPELINE
# ============================================================================

def main():
    print("="*60)
    print("ENERGY CONSUMPTION FORECASTING - TRAINING PIPELINE")
    print("="*60)

    set_seed(RANDOM_SEED)
    print(f"\nRandom seed set to {RANDOM_SEED}")

    df = load_data(DATA_PATH)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), x_scaler, y_scaler = prepare_data(df)
    print("\nData Loaded")

    train_dataset = TensorDataset(
        torch.FloatTensor(x_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(x_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(
        input_size = 8,
        hidden_size = HIDDEN_SIZE,
        num_layers = NUM_LAYERS,
        output_size = 1
    )
    model.to(device)

    print(f"Model initialized (hidden size = {HIDDEN_SIZE}, (num layers = {NUM_LAYERS}))")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}", end="")
        
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0

            Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        else:
            patience_counter += 1
            print(f"patience : {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter > EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch : {epoch + 1}")
                break

    print("-"*60)
    print(f"✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Total epochs: {epoch+1}")

    Path(X_SCALER_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(X_SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(x_scaler, f)
    Path(Y_SCALER_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(Y_SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(y_scaler, f)

    print("Best Model and Scaler saved")
    print("TRAINING COMPLETE")

if __name__ == '__main__':
    main()