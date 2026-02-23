# Energy Consumption Forecasting with LSTM

A multi-version LSTM project for household energy consumption forecasting, built with PyTorch. Tracks the full progression from single-step to multi-step prediction with systematic evaluation against baselines.

---

## Motivation

Predicting household energy consumption is a practical time series problem with real applications in smart grid management and demand forecasting. This project explores how well an LSTM can learn temporal patterns in energy data compared to simple statistical baselines.

---

## Dataset

**UCI Household Power Consumption**: minute-level electricity measurements from a single household over 4 years (~2 million records).

- Target: `Global_active_power` (kW)
- Sampling: minute-level, aggregated to hourly for modeling
- Split: 70% train / 15% val / 15% test (chronological, no shuffling)

---

## Project Progression

### V1: Single Feature, Single Step
- Input: global active power only
- Output: next hour prediction
- Establishes baseline LSTM performance

### V2: Multi-Feature, Single Step
- Added time features: hour, day, month (sin/cos encoded), weekend flag
- Sequence length: 168 hours (one week lookback)
- Marginal improvement over V1; time features help but dataset is strongly autocorrelated

### V3: Multi-Feature, Multi-Step (24h horizon)
- Direct multi-step prediction: one forward pass outputs 24 future values simultaneously
- Key finding: naive baseline is extremely competitive on this dataset due to high autocorrelation
- Performance degrades gracefully at longer horizons as expected

---

## Architecture

```
Input: (batch, 168, 8) -- 168 hour sequence, 8 features
         ↓
LSTM: hidden_size=96, num_layers=1, dropout=0.2
         ↓
Linear: hidden_size → output_size (1 or 24)
         ↓
Output: (batch, output_size)
```

**Features:**
- `Global_active_power`: target variable (also used as input feature)
- `hour_sin`, `hour_cos`: cyclical hour encoding
- `day_sin`, `day_cos`: cyclical day of week encoding
- `month_sin`, `month_cos`: cyclical month encoding
- `weekend`: binary flag

---

## Results

### V3: 24-Hour Multi-Step Prediction

| Model | MAE (kW) | RMSE (kW) | MAPE (%) |
|---|---|---|---|
| Naive (t-1 repeated) | 0.3063 | 0.6045 | 41.46 |
| Seasonal Naive (t-24) | 0.4362 | 0.7596 | 60.07 |
| **LSTM (V3)** | **0.3104** | **0.5425** | **44.50** |

The LSTM beats seasonal naive by a significant margin and matches naive (t-1) on MAE while achieving lower RMSE, indicating better handling of large errors.

### Key Finding

The naive baseline is remarkably strong on this dataset. Energy consumption is highly autocorrelated, meaning yesterday's value is a good predictor of today's. The LSTM's advantage is more pronounced at longer horizons where simple autocorrelation weakens.

---

## Evaluation Metrics

Beyond MAE/RMSE, the pipeline tracks:

- **MAPE**: percentage error, scale-independent
- **R² Score**: variance explained by the model
- **Directional Accuracy**: does the model predict the right trend direction?
- **Peak Detection Precision/Recall**: how well does the model identify high consumption events (top 10th percentile)?

---

## Stack

| Component | Choice |
|---|---|
| Framework | PyTorch |
| Data | pandas, numpy |
| Scaling | scikit-learn MinMaxScaler |
| Evaluation | scikit-learn metrics |

---

## Project Structure

```
Energy-Consumption/
├── data/
│   └── household_power_consumption.txt
├── models/
│   └── v2.0/
│       ├── best_model.pth
│       ├── x_scaler.pkl
│       └── y_scaler.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
├── train_pipeline.py
├── evaluation_pipeline.py
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Joltsy10/Energy-Consumptiom
cd Energy-Consumptiom
pip install torch pandas numpy scikit-learn
```

Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place it in `data/`.

**Train:**
```bash
python train_pipeline.py
```

**Evaluate:**
```bash
python evaluation_pipeline.py
```

---

## Design Decisions

**Why direct multi-step over autoregressive?** Direct prediction outputs all 24 steps in one forward pass rather than feeding predictions back iteratively. This avoids error accumulation; in autoregressive decoding a bad step 1 prediction degrades every subsequent step. For a 24-step horizon, direct prediction is simpler and more stable.

**Why MinMaxScaler over StandardScaler?** Energy consumption has a bounded range and no strong Gaussian assumption. MinMaxScaler keeps values in [0,1] which suits LSTM sigmoid and tanh activations well.

**Why sin/cos encoding for time features?** Cyclical encoding preserves the circular nature of time. Hour 23 and hour 0 should be close, not far apart. Simple integer encoding would mislead the model.

---

## What I Learned

- High autocorrelation datasets make naive baselines hard to beat; this is a property of the data, not a failure of the model
- Multi-step prediction is significantly harder than single-step; performance degrades at longer horizons as expected
- Early stopping based on val loss is critical; the model overfits quickly on this dataset
- Config files should be set up from day one; changing hyperparameters across multiple files manually is error-prone

---

## Roadmap

- Per-horizon evaluation showing step-by-step MAE degradation
- Autoregressive decoding as comparison to direct multi-step
- Transformer-based alternative (temporal fusion transformer)
- Multi-household generalization