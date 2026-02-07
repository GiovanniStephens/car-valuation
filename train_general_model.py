#!/usr/bin/env python3
"""
Train a general cross-car-type model on combined data from all car types.

This model is used for blending with car-specific models when the car-specific
training set is small (see predict_blended() in inference.py).

Usage:
    python train_general_model.py
"""

import os
import glob
import datetime
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    from utils import load_config, load_training_config
    from feature_engineering import feature_engineering

    config = load_config()
    training_config = load_training_config()

    with open("model_settings.yml", "r") as f:
        model_config = yaml.load(f, Loader=SafeLoader)
    general_config = model_config.get("general_model", {})
    time_limit = general_config.get("time_limit", 1800)

    # Find all data files
    data_files = glob.glob("data/*_data.pkl")
    if not data_files:
        print("ERROR: No data files found in data/")
        return

    print(f"\n{'='*60}")
    print("  TRAINING GENERAL CROSS-CAR-TYPE MODEL")
    print(f"{'='*60}")
    print(f"  Data files found: {len(data_files)}\n")

    all_data = []
    sample_counts = {}

    for data_file in data_files:
        # Extract make_model from filename: data/Make_Model_data.pkl
        basename = os.path.basename(data_file)
        parts = basename.replace("_data.pkl", "").split("_", 1)
        if len(parts) != 2:
            print(f"  Skipping {basename} (unexpected filename format)")
            continue

        make, model_name = parts[0], parts[1]
        print(f"  Loading {make} {model_name}...")

        try:
            data = pd.read_pickle(data_file)

            # Remove auction items
            try:
                data = data[data["BidCount"].isnull()].reset_index()
            except KeyError:
                pass

            if len(data) < 10:
                print(f"    Skipping (only {len(data)} listings)")
                continue

            # Update config with make/model for classifier lookup
            config_for_car = config.copy()
            config_for_car["make"] = make
            config_for_car["model"] = model_name
            data = feature_engineering(data, config_for_car)
            data = data.drop("ListingId", axis=1, errors="ignore")

            # Add Make and Model columns for the general model
            data["Make"] = make
            data["Model"] = model_name

            sample_counts[f"{make}_{model_name}"] = len(data)
            all_data.append(data)
            print(f"    Added {len(data)} samples")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    if not all_data:
        print("\nERROR: No valid data loaded")
        return

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total combined samples: {len(combined)}")
    print(f"  Car types: {len(all_data)}")

    # Ensure categorical types
    for col in ["Make", "Model", "Region", "Transmission", "Fuel", "Cylinders", "colour_label", "stereo_label"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(str)

    # Train
    model_path = "models/general_model"
    print(f"\n  Training AutoGluon general model...")
    print(f"  Preset: {training_config.get('preset', 'best_quality')}")
    print(f"  Time limit: {time_limit}s\n")

    predictor = TabularPredictor(
        label="price",
        path=model_path,
        eval_metric=training_config.get("eval_metric", "mean_absolute_error"),
    )
    predictor.fit(
        train_data=combined,
        presets=training_config.get("preset", "best_quality"),
        time_limit=time_limit,
        num_bag_folds=training_config.get("num_bag_folds", 5),
    )

    # Save metadata
    metadata = {
        "n_training_samples": len(combined),
        "training_date": datetime.datetime.now().isoformat(),
        "car_types": len(all_data),
        "sample_counts": sample_counts,
    }
    metadata_path = os.path.join(model_path, "metadata.yml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # Show leaderboard
    print("\nModel leaderboard:")
    print(predictor.leaderboard(silent=True).to_string())

    print(f"\nGeneral model saved to {model_path}")
    print(f"Total samples: {len(combined)}, Car types: {len(all_data)}")


if __name__ == "__main__":
    main()
