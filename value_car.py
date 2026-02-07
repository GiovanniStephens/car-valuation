#!/usr/bin/env python3
"""
Unified car valuation script.

Reads car_to_value.yml, ensures data and model exist (fetching/training if needed),
then outputs the valuation.

Usage:
    python value_car.py
    python value_car.py --dashboard  # Launch Streamlit dashboard
"""

import os
import sys
import yaml
from yaml.loader import SafeLoader


def load_car_to_value():
    """Load the car specifications from car_to_value.yml."""
    with open("car_to_value.yml", "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def check_data_exists(make, model):
    """Check if data file exists for the given make/model."""
    from utils import get_data_path

    return os.path.exists(get_data_path(make, model))


def check_model_exists(make, model):
    """Check if trained AutoGluon model exists for the given make/model."""
    from utils import get_model_path

    return os.path.isdir(get_model_path(make, model))


def update_car_search_config(make, model):
    """Update car_search.yml with the make/model from car_to_value.yml."""
    with open("car_search.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    config["make"] = make
    config["model"] = model

    with open("car_search.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated car_search.yml with make={make}, model={model}")


def fetch_data(make, model):
    """Fetch data from TradeMe API."""
    from utils import fetch_trademe_data

    print(f"\n{'=' * 50}")
    print(f"  Fetching data for {make} {model}")
    print(f"{'=' * 50}\n")

    # Ensure car_search.yml is updated
    update_car_search_config(make, model)

    # Fetch data using the consolidated utility function
    fetch_trademe_data(make, model)


def train_model(make, model):
    """Train the valuation model using AutoGluon."""
    print(f"\n{'=' * 50}")
    print(f"  Training model for {make} {model}")
    print(f"{'=' * 50}\n")

    # Ensure car_search.yml is updated
    update_car_search_config(make, model)

    # Import and run model training
    from valuation_model import train_model_cli

    train_model_cli()

    print(f"\nModel training complete for {make} {model}")


def run_valuation():
    """Run the valuation using car_to_value.yml."""
    print(f"\n{'=' * 50}")
    print("  Running Valuation")
    print(f"{'=' * 50}\n")

    from inference import value_car

    value_car()


def main():
    """Main entry point - orchestrates data fetch, training, and valuation."""
    # Load car to value
    car_specs = load_car_to_value()
    make = car_specs["make"]
    model = car_specs["model"]

    print(f"\n{'=' * 50}")
    print("  Car Valuation Pipeline")
    print(f"  {make} {model}")
    print(f"{'=' * 50}")

    # Step 1: Check/fetch data
    if check_data_exists(make, model):
        print(f"\n[OK] Data exists: data/{make}_{model}_data.pkl")
    else:
        print(f"\n[!] Data not found for {make} {model}")
        fetch_data(make, model)

    # Step 2: Check/train model
    if check_model_exists(make, model):
        print(f"[OK] Model exists: models/{make}_{model}/")
    else:
        print(f"\n[!] Model not found for {make} {model}")
        train_model(make, model)

    # Step 3: Run valuation
    run_valuation()


if __name__ == "__main__":
    if "--dashboard" in sys.argv:
        import subprocess

        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    else:
        main()
