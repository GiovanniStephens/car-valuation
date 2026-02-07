# Car Valuation

Machine learning-based car valuation system using TradeMe NZ listings data.

## Project Structure

### Core Modules

| Module | Description |
|--------|-------------|
| `feature_engineering.py` | Transform raw data into model features (colour/stereo classification, derived features) |
| `training.py` | Train AutoGluon models (main, quantile, tail) |
| `inference.py` | Make predictions and valuations (blending, confidence intervals, depreciation) |
| `utils.py` | Shared utilities (config loading, data fetching, model metadata, filters) |

### Entry Points (Scripts)

| Script | Description | Usage |
|--------|-------------|-------|
| `value_car.py` | Single car: fetch data → train → value | `python value_car.py` |
| `find_undervalued.py` | Batch search for undervalued cars | `python find_undervalued.py [--fetch] [--train]` |
| `dashboard.py` | Interactive web UI (Streamlit) | `streamlit run dashboard.py` |
| `train_general_model.py` | Train cross-car general model | `python train_general_model.py` |
| `valuation_model.py` | CLI for training/valuation | `python valuation_model.py [value\|interactive]` |

### Config Files

| File | Description |
|------|-------------|
| `model_settings.yml` | Training hyperparameters, embedding model, general model settings |
| `car_search.yml` | Make/model for single-car operations |
| `car_to_value.yml` | Specific car specs to value |
| `undervalued_search.yml` | Multi-car batch search config |

## Quick Start

### 1. Set up TradeMe API credentials

```python
import keyring
keyring.set_password("Trademe", "key", "YOUR_CONSUMER_KEY")
keyring.set_password("Trademe", "secret", "YOUR_CONSUMER_SECRET")
```

### 2. Configure your car

Edit `car_to_value.yml`:

```yaml
make: Toyota
model: Rav4
region: Auckland
year: 2018
engine_size: 2.5L
odometer: 80000
fuel_type: Petrol
transmission: Automatic
cylinders: 4
is_4wd: true
```

### 3. Run valuation

```bash
python value_car.py
```

This will:
1. Fetch data from TradeMe (if not already present)
2. Train a model (if not already trained)
3. Output the valuation with confidence intervals

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

## Architecture

### Model Types

- **Car-specific model**: AutoGluon ensemble trained on one make/model
- **Quantile model**: Provides confidence intervals (10th, 25th, 50th, 75th, 90th percentiles)
- **Tail models**: Ridge regression for extreme values (low/high price tails)
- **General model**: Cross-car-type model for blending when data is limited

### Blending Strategy

When making predictions, the system blends car-specific and general models:
- n < 50 samples: General model only
- 50 ≤ n < 200: Linear blend
- n ≥ 200: Car-specific only

### Feature Engineering

Raw data is transformed with:
- Colour/stereo classification via sentence embeddings + PyCaret classifiers
- Derived features: `Age = 2026 - Year`, `KmPerYear = Odometer / max(Age, 1)`
- Categorical columns kept as strings (AutoGluon handles them natively)

## Model Storage

```
models/
├── Toyota_Rav4/              # Car-specific model
│   ├── predictor.pkl
│   ├── metadata.yml
│   ├── tail_info.pkl
│   └── ...
├── Toyota_Rav4_quantile/     # Quantile model
│   └── ...
└── general_model/            # Cross-car-type model
    └── ...
```

## Dependencies

Key packages:
- `autogluon` - AutoML framework
- `pycaret` - For colour/stereo classifiers
- `sentence-transformers` - Text embeddings
- `streamlit` - Dashboard UI
- `requests-oauthlib` - TradeMe API
