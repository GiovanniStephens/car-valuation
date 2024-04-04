# Valuation Model

This repository contains a Python script for a valuation model using the PyCaret library. The model is a Random Forest regression model, which is trained, tuned, and finalized on a given dataset. The model is then used to make predictions on out-of-sample data and a specific data point.

## Code Overview

The script begins by setting up a regression model with specific parameters. The target variable is 'price' and the training data size is set to 80% of the total data. The model does not normalize the data, apply any transformations, or remove outliers. The data is shuffled before splitting and the model uses a k-fold cross-validation strategy with 5 folds.

```python
regression = setup(data=data,
                   target='price',
                   train_size=0.8,
                   session_id=42,
                   normalize=False,
                   transformation=False,
                   transform_target=False,
                   remove_outliers=False,
                   outliers_threshold=0.05,
                   data_split_shuffle=True,
                   fold_strategy='kfold',
                   fold=5,
                   fold_shuffle=True,
                   fold_groups=None,
                   n_jobs=-1,
                   use_gpu=False,
                   custom_pipeline=None)
```

A Random Forest model is created, tuned, and finalized. The finalized model is saved as 'final_rf_model'.

```python
rf = create_model('rf')
tuned_rf = tune_model(rf)
final_rf = finalize_model(tuned_rf)
save_model(final_rf, 'final_rf_model')
```

The model is used to make predictions on out-of-sample data and a specific data point (TradeMe listing ID defined in the config). The predictions are printed to the console.

```python
predictions = predict_model(final_rf, data=out_of_sample)
valuation = predict_model(final_rf, data=data_point_to_valuate)
print(valuation)
```

Finally, a residuals plot of the model is generated.

```python
plot_model(final_rf, plot='residuals')
```

## Requirements

- Python 3.6+
- PyCaret
- Pandas
- Umap-learn
- hdbscan
- sentence-transformers
- requests_oauthlib
- keyring

## Usage

1. Clone the repository.
2. Install the requirements.
3. Run the `valuation_model.py` script.

# get_data Module

The `get_data` module is responsible for fetching and preparing the data that will be used by the valuation model. 

## Overview

This module contains functions to load the data from a specified source, clean the data by handling missing values and outliers, and preprocess the data by performing necessary transformations and encoding categorical variables.

## Usage

To use this module, run the module as a script with the following command:

```bash
python get_data.py
```

The saved Pickle data can then be used as input to the valuation model.

## Requirements

This module requires the following Python packages:

- pandas
- numpy
