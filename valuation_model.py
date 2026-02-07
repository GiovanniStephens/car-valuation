"""
Car valuation model - main CLI and backward-compatible import hub.

This module serves two purposes:
1. Provides backward-compatible imports for existing code
2. Provides the CLI entry point for training and valuation

Usage:
    python valuation_model.py              # Train model
    python valuation_model.py value        # Value car from config
    python valuation_model.py interactive  # Interactive mode
"""

import os
import sys
import pickle
import warnings

# Disable tokenizer parallelism before forking to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# =============================================================================
# Backward-compatible imports
# =============================================================================
# These re-exports maintain compatibility with existing code that imports from
# valuation_model (e.g., `from valuation_model import feature_engineering`)

from utils import (
    load_config as read_config,
    load_training_config,
    load_data,
    save_model_metadata,
    load_model_metadata,
    apply_filters,
    get_model_path,
    get_quantile_model_path,
)

from feature_engineering import (
    feature_engineering,
    build_data_point,
    classify_colour,
    classify_stereo,
)

from training import (
    train_autogluon,
    train_quantile_model,
    train_tail_models,
    save_tail_models,
    load_tail_models,
)

from inference import (
    predict_blended,
    predict_with_confidence,
    calculate_depreciation,
    identify_undervalued_cars,
    get_input_with_match,
    get_numeric_input,
    run_interactive_valuation as value_input_car,
    value_car as value_from_config,
)


# =============================================================================
# CLI Entry Points
# =============================================================================


def train_model_cli():
    """Train a model using settings from car_search.yml and model_settings.yml.

    This is the main training workflow that:
    1. Loads data for the configured make/model
    2. Applies feature engineering
    3. Trains the main AutoGluon model
    4. Trains the quantile model for confidence intervals
    5. Trains tail models for extreme value handling
    6. Evaluates on out-of-sample data
    7. Identifies undervalued cars in the dataset
    """
    from autogluon.tabular import TabularPredictor
    import pandas as pd

    config = read_config()
    make = config["make"]
    model_name = config["model"]
    training_config = load_training_config()

    data = load_data(f"data/{make}_{model_name}_data.pkl")
    try:
        data = data[data["BidCount"].isnull()].reset_index()
    except KeyError:
        pass

    data = feature_engineering(data, config)
    data_point_to_valuate = data[data["ListingId"] == config["valuation_listing_id"]]
    listing_ids = data["ListingId"]
    data = data.drop(data_point_to_valuate.index)
    data = data.drop("ListingId", axis=1)

    out_of_sample = data.sample(frac=0.4, random_state=42)
    data = data.drop(out_of_sample.index)

    model_path = get_model_path(make, model_name)

    # Train main point-estimate model
    print(f"\nTraining AutoGluon model for {make} {model_name}...")
    print(f"  Training samples: {len(data)}")
    print(f"  Preset: {training_config.get('preset', 'best_quality')}")
    print(f"  Time limit: {training_config.get('time_limit', 600)}s\n")

    predictor, n_samples = train_autogluon(data, model_path, training_config)
    save_model_metadata(model_path, n_samples, make, model_name)

    # Train quantile model for confidence intervals
    print("\nTraining quantile model...")
    quantile_path = get_quantile_model_path(make, model_name)
    train_quantile_model(data, quantile_path, training_config)

    # Train tail models
    print("Training tail regression models...")
    tail_info = train_tail_models(data, training_config)
    save_tail_models(tail_info, model_path)
    print(f"  Low tail samples: {tail_info['low_n']}, model: {'yes' if tail_info['low_model'] else 'no'}")
    print(f"  High tail samples: {tail_info['high_n']}, model: {'yes' if tail_info['high_model'] else 'no'}")

    # Evaluate on out-of-sample data
    print("\nModel leaderboard:")
    print(predictor.leaderboard(data=out_of_sample, silent=True).to_string())

    # Valuate specific car if requested
    if data_point_to_valuate.shape[0] > 0:
        valuation_data = data_point_to_valuate.drop("ListingId", axis=1, errors="ignore")
        pred = predictor.predict(valuation_data)
        print(f"\nValuation for listing {config['valuation_listing_id']}: ${pred.iloc[0]:,.0f}")

    # Identify undervalued cars
    full_data = pd.concat([data, out_of_sample])
    if data_point_to_valuate.shape[0] > 0:
        valuation_data_full = data_point_to_valuate.drop("ListingId", axis=1, errors="ignore")
        full_data = pd.concat([full_data, valuation_data_full])
    identify_undervalued_cars(full_data, config, listing_ids, predictor)


def print_cli_usage():
    """Print CLI usage instructions."""
    print("""
Usage: python valuation_model.py [command]

Commands:
  (no args)     Train a new model using car_search.yml settings
  value         Value a car from car_to_value.yml config file
  interactive   Interactively enter car details for valuation

Examples:
  python valuation_model.py              # Train model
  python valuation_model.py value        # Value car from config
  python valuation_model.py interactive  # Interactive mode
""")


# Backward-compatible aliases
main = train_model_cli
print_usage = print_cli_usage


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_model_cli()
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()
        if command == "value":
            value_from_config()
        elif command == "interactive":
            value_input_car()
        elif command in ["-h", "--help", "help"]:
            print_cli_usage()
        else:
            print(f"Unknown command: {command}")
            print_cli_usage()
            sys.exit(1)
    else:
        print("Error: too many arguments")
        print_cli_usage()
        sys.exit(1)
