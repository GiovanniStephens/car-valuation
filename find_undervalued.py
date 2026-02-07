#!/usr/bin/env python3
"""
Find undervalued cars across multiple makes/models.

Usage:
    python find_undervalued.py              # Use existing data and models
    python find_undervalued.py --fetch      # Fetch fresh data first
    python find_undervalued.py --train      # Retrain models
    python find_undervalued.py --fetch --train  # Full refresh
    python find_undervalued.py --train-general  # Also train general model
"""

import os
import sys
import argparse
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Disable tokenizer parallelism before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_undervalued_config():
    """Load undervalued search config."""
    with open("undervalued_search.yml", "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def fetch_data(make, model):
    """Fetch data for a single make/model from TradeMe."""
    from utils import fetch_trademe_data

    print(f"  Fetching {make} {model}...")

    try:
        fetch_trademe_data(make, model)
        return True
    except SystemExit:
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def train_model(make, model):
    """Train model for a single make/model using AutoGluon."""
    from utils import load_config, load_training_config, load_data, get_model_path, save_model_metadata
    from feature_engineering import feature_engineering
    from training import train_autogluon, train_quantile_model, train_tail_models, save_tail_models

    print(f"  Training {make} {model}...")

    try:
        config = load_config()
        training_config = load_training_config()
        data = load_data(f"data/{make}_{model}_data.pkl")

        # Remove auction items
        try:
            data = data[data["BidCount"].isnull()].reset_index()
        except KeyError:
            pass

        if len(data) < 20:
            print(f"    Insufficient data ({len(data)} listings)")
            return False

        data = feature_engineering(data, config)
        data = data.drop("ListingId", axis=1, errors='ignore')

        print(f"    Data shape: {data.shape}")

        model_path = get_model_path(make, model)

        # Train main model
        predictor, n_samples = train_autogluon(data, model_path, training_config)
        save_model_metadata(model_path, n_samples, make, model)

        # Train quantile model
        quantile_path = f"{model_path}_quantile"
        train_quantile_model(data, quantile_path, training_config)

        # Train tail models
        tail_info = train_tail_models(data, training_config)
        save_tail_models(tail_info, model_path)

        print(f"    Trained successfully (n={n_samples})")
        return True

    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_undervalued(make, model, min_discount_pct, max_discount_pct, filters, comparison_region=None):
    """Find undervalued cars for a single make/model."""
    from autogluon.tabular import TabularPredictor
    from utils import load_config, load_data, get_model_path, load_model_metadata
    from feature_engineering import feature_engineering

    try:
        # Check for AutoGluon model
        model_path = get_model_path(make, model)
        if not os.path.isdir(model_path):
            print(f"  {make} {model}: No model found")
            return None

        # Load model
        predictor = TabularPredictor.load(model_path)

        # Load and prepare data
        config = load_config()
        data = load_data(f"data/{make}_{model}_data.pkl")

        # Remove auction items
        try:
            data = data[data["BidCount"].isnull()].reset_index()
        except KeyError:
            pass

        # Keep listing IDs for output
        listing_ids = data[["ListingId"]].copy()
        if "Title" in data.columns:
            listing_ids["Title"] = data["Title"]
        if "PriceDisplay" in data.columns:
            listing_ids["PriceDisplay"] = data["PriceDisplay"]
        if "Region" in data.columns:
            listing_ids["OrigRegion"] = data["Region"]

        data = feature_engineering(data, config)
        data_listing_ids = data["ListingId"]
        data = data.drop("ListingId", axis=1)

        # Predict
        predictions = predictor.predict(data)
        data = data.copy()
        data["prediction_label"] = predictions

        # Calculate discount percentage
        data["discount_pct"] = (
            (data["prediction_label"] - data["price"])
            / data["prediction_label"] * 100
        )

        # Regional comparison if specified
        if comparison_region and "Region" in data.columns:
            comparison_data = data.copy()
            comparison_data["Region"] = comparison_region
            comparison_predictions = predictor.predict(comparison_data)
            data["comparison_value"] = comparison_predictions
            data["regional_diff"] = data["comparison_value"] - data["prediction_label"]
            data["regional_diff_pct"] = (
                data["regional_diff"] / data["prediction_label"] * 100
            )

        # Filter by percentage thresholds
        undervalued = data[
            (data["discount_pct"] >= min_discount_pct) &
            (data["discount_pct"] <= max_discount_pct)
        ].copy()

        if len(undervalued) == 0:
            print(f"  {make} {model}: No undervalued cars found")
            return None

        # Add listing info
        undervalued["ListingId"] = data_listing_ids.loc[undervalued.index].values
        undervalued = undervalued.merge(listing_ids, on="ListingId", how="left")

        # Add make/model
        undervalued["Make"] = make
        undervalued["Model"] = model

        # Apply additional filters
        for filter_rule in filters or []:
            if filter_rule is None:
                continue
            column = filter_rule.get("column")
            operator = filter_rule.get("operator")
            value = filter_rule.get("value")
            if column and operator and value is not None:
                if column in undervalued.columns:
                    undervalued = undervalued.query(f"{column} {operator} @value")

        print(f"  {make} {model}: Found {len(undervalued)} undervalued")
        return undervalued

    except Exception as e:
        print(f"  {make} {model}: Error - {e}")
        return None


def train_general_model(cars):
    """Train the general cross-car-type model."""
    print("\nTraining general model...")
    from train_general_model import main as train_general_main
    train_general_main()


def main():
    parser = argparse.ArgumentParser(description="Find undervalued cars")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data")
    parser.add_argument("--train", action="store_true", help="Retrain models")
    parser.add_argument("--train-general", action="store_true", help="Also train general cross-car model")
    args = parser.parse_args()

    config = load_undervalued_config()
    cars = config["cars"]
    min_discount = config["minimum_discount_percent"]
    max_discount = config["maximum_discount_percent"]
    filters = config.get("filters", [])
    comparison_region = config.get("comparison_region")

    print("\n" + "=" * 60)
    print("  UNDERVALUED CAR SEARCH")
    print("=" * 60)
    print(f"  Discount range: {min_discount}% - {max_discount}%")
    print(f"  Cars to search: {len(cars)}")
    if comparison_region:
        print(f"  Comparison region: {comparison_region}")
    print("=" * 60)

    # Fetch data if requested
    if args.fetch:
        print("\nFetching data...")
        for car in cars:
            fetch_data(car["make"], car["model"])

    # Train models if requested
    if args.train:
        print("\nTraining models...")
        for car in cars:
            train_model(car["make"], car["model"])

    # Train general model if requested
    if args.train_general:
        train_general_model(cars)

    # Find undervalued cars
    print("\nSearching for undervalued cars...")
    all_undervalued = []

    for car in cars:
        result = find_undervalued(
            car["make"], car["model"],
            min_discount, max_discount, filters, comparison_region
        )
        if result is not None and len(result) > 0:
            all_undervalued.append(result)

    if not all_undervalued:
        print("\nNo undervalued cars found matching criteria.")
        return

    # Combine results
    combined = pd.concat(all_undervalued, ignore_index=True)
    combined = combined.sort_values("discount_pct", ascending=False)

    # Display results
    print("\n" + "=" * 60)
    print(f"  RESULTS: {len(combined)} UNDERVALUED CARS")
    print("=" * 60 + "\n")

    display_cols = ["Make", "Model", "price", "prediction_label", "discount_pct"]
    if "Title" in combined.columns:
        display_cols.insert(2, "Title")
    if "Region" in combined.columns:
        display_cols.append("Region")
    elif "OrigRegion" in combined.columns:
        display_cols.append("OrigRegion")
    # Add regional comparison columns if present
    if "comparison_value" in combined.columns and comparison_region:
        display_cols.append("comparison_value")
        display_cols.append("regional_diff_pct")
    if "ListingId" in combined.columns:
        display_cols.append("ListingId")

    # Format for display
    available_cols = [c for c in display_cols if c in combined.columns]
    display_df = combined[available_cols].copy()
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:,.0f}")
    display_df["prediction_label"] = display_df["prediction_label"].apply(lambda x: f"${x:,.0f}")
    display_df["discount_pct"] = display_df["discount_pct"].apply(lambda x: f"{x:.1f}%")
    if "comparison_value" in display_df.columns:
        display_df["comparison_value"] = display_df["comparison_value"].apply(lambda x: f"${x:,.0f}")
        display_df["regional_diff_pct"] = display_df["regional_diff_pct"].apply(lambda x: f"{x:+.1f}%")
    col_renames = {
        "prediction_label": "Est. Value",
        "discount_pct": "Discount",
        "comparison_value": f"{comparison_region} Value" if comparison_region else "Comp. Value",
        "regional_diff_pct": "Region Diff",
        "OrigRegion": "Region",
    }
    display_df.columns = [col_renames.get(c, c) for c in display_df.columns]

    print(display_df.to_string(index=False))

    # Print TradeMe URLs
    if "ListingId" in combined.columns:
        print("\n" + "-" * 60)
        print("  LISTING URLS")
        print("-" * 60)
        for _, row in combined.head(10).iterrows():
            print(f"  https://www.trademe.co.nz/a/{row['ListingId']}")

    print()


if __name__ == "__main__":
    main()
