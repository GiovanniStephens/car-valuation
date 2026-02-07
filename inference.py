"""
Inference module for car valuation.

This module handles prediction and valuation:
- Blended predictions (car-specific + general model)
- Confidence intervals from quantile models
- Tail-adjusted predictions
- Depreciation forecasting
- Interactive and config-based valuation workflows
"""

import os
import pickle
import pandas as pd
import numpy as np
import yaml
from yaml.loader import SafeLoader
from autogluon.tabular import TabularPredictor
from fuzzywuzzy import process

from utils import load_config, load_model_metadata, get_model_path, get_quantile_model_path


def predict_blended(data_point: pd.DataFrame, make: str, model_name: str) -> float:
    """Predict using blended car-specific + general model.

    Blending weight depends on car-specific training sample size:
    - n < 50:       general model only
    - 50 <= n < 200: linear ramp
    - n >= 200:     car-specific only

    Args:
        data_point: Single-row DataFrame with feature columns.
        make: Car make.
        model_name: Car model.

    Returns:
        Blended price prediction.

    Raises:
        ValueError: If no model is available for the given make/model.
    """
    with open("model_settings.yml", "r") as f:
        model_config = yaml.load(f, Loader=SafeLoader)
    general_config = model_config.get("general_model", {})

    if not general_config.get("enabled", False):
        # General model disabled -- use car-specific only
        specific_path = get_model_path(make, model_name)
        predictor = TabularPredictor.load(specific_path)
        return predictor.predict(data_point).iloc[0]

    blend_min = general_config.get("blend_threshold_min", 50)
    blend_max = general_config.get("blend_threshold_max", 200)

    # Car-specific prediction
    specific_path = get_model_path(make, model_name)
    specific_pred = None
    n_samples = 0

    if os.path.isdir(specific_path):
        meta = load_model_metadata(specific_path)
        n_samples = meta.get("n_training_samples", 0)
        predictor = TabularPredictor.load(specific_path)
        specific_pred = predictor.predict(data_point).iloc[0]

    # General model prediction
    general_path = "models/general_model"
    general_pred = None

    if os.path.isdir(general_path):
        general_predictor = TabularPredictor.load(general_path)
        general_data = data_point.copy()
        general_data["Make"] = make
        general_data["Model"] = model_name
        general_pred = general_predictor.predict(general_data).iloc[0]

    # Blend
    if specific_pred is None and general_pred is None:
        raise ValueError(f"No model available for {make} {model_name}")
    if specific_pred is None:
        return general_pred
    if general_pred is None:
        return specific_pred

    if n_samples < blend_min:
        weight_specific = 0.0
    elif n_samples >= blend_max:
        weight_specific = 1.0
    else:
        weight_specific = (n_samples - blend_min) / (blend_max - blend_min)

    return weight_specific * specific_pred + (1 - weight_specific) * general_pred


def predict_with_confidence(
    data_point: pd.DataFrame, make: str, model_name: str
) -> dict:
    """Predict with confidence intervals from the quantile model.

    Also applies tail blending if the point estimate falls in a tail region.

    Args:
        data_point: Single-row DataFrame with feature columns.
        make: Car make.
        model_name: Car model.

    Returns:
        Dictionary with:
            - estimate: Point estimate (possibly tail-adjusted)
            - ci_80: Tuple of (low, high) for 80% confidence interval (if available)
            - ci_50: Tuple of (low, high) for 50% confidence interval (if available)
            - tail: "low" or "high" if tail adjustment was applied
    """
    point_estimate = predict_blended(data_point, make, model_name)
    specific_path = get_model_path(make, model_name)

    result = {"estimate": point_estimate}

    # Quantile predictions for confidence intervals
    quantile_path = get_quantile_model_path(make, model_name)
    if os.path.isfile(os.path.join(quantile_path, "predictor.pkl")):
        q_predictor = TabularPredictor.load(quantile_path)
        q_preds = q_predictor.predict(data_point)
        if isinstance(q_preds, pd.DataFrame):
            result["ci_80"] = (q_preds.iloc[0, 0], q_preds.iloc[0, 4])  # 0.1, 0.9
            result["ci_50"] = (q_preds.iloc[0, 1], q_preds.iloc[0, 3])  # 0.25, 0.75

    # Tail blending
    tail_path = os.path.join(specific_path, "tail_info.pkl")
    if os.path.exists(tail_path):
        with open(tail_path, "rb") as f:
            tail_info = pickle.load(f)

        q25 = tail_info["q25"]
        q75 = tail_info["q75"]
        feature_cols = tail_info["feature_cols"]

        numeric_point = (
            data_point[feature_cols].fillna(0)
            if all(c in data_point.columns for c in feature_cols)
            else None
        )

        if numeric_point is not None:
            if point_estimate < q25 and tail_info.get("low_model") is not None:
                tail_pred = tail_info["low_model"].predict(numeric_point)[0]
                # Blend: 50/50 between main and tail model in tail region
                result["estimate"] = 0.5 * point_estimate + 0.5 * tail_pred
                result["tail"] = "low"
            elif point_estimate > q75 and tail_info.get("high_model") is not None:
                tail_pred = tail_info["high_model"].predict(numeric_point)[0]
                result["estimate"] = 0.5 * point_estimate + 0.5 * tail_pred
                result["tail"] = "high"

    return result


def calculate_depreciation(
    predictor: TabularPredictor, data_point: pd.DataFrame, predicted_price: float
) -> dict:
    """Calculate 12-month depreciation estimates using AutoGluon predictor.

    Uses the predictor to estimate value at future states (age+1 year, with
    different mileage scenarios).

    Args:
        predictor: Trained AutoGluon TabularPredictor.
        data_point: Single-row DataFrame with current car features.
        predicted_price: Current predicted price.

    Returns:
        Dictionary with:
            - value_10k: Predicted value after 1 year and 10,000 km
            - value_0k: Predicted value after 1 year with minimal travel
            - depreciation_10k: Price drop with 10,000 km travel
            - depreciation_0k: Price drop with minimal travel
            - reliable: True if depreciation estimates seem reasonable
    """
    # Scenario 1: +1 year age, +10,000 km (normal usage)
    future_10k = data_point.copy()
    future_10k["Year"] = future_10k["Year"] - 1
    future_10k["Odometer"] = future_10k["Odometer"] + 10000
    future_10k["Age"] = future_10k["Age"] + 1
    future_10k["KmPerYear"] = future_10k["Odometer"] / future_10k["Age"].clip(lower=1)
    value_10k = predictor.predict(future_10k).iloc[0]

    # Scenario 2: +1 year age, +0 km (parked)
    future_0k = data_point.copy()
    future_0k["Year"] = future_0k["Year"] - 1
    future_0k["Age"] = future_0k["Age"] + 1
    future_0k["KmPerYear"] = future_0k["Odometer"] / future_0k["Age"].clip(lower=1)
    value_0k = predictor.predict(future_0k).iloc[0]

    depreciation_10k = predicted_price - value_10k
    depreciation_0k = predicted_price - value_0k
    reliable = depreciation_10k > 0 and depreciation_0k > 0

    return {
        "value_10k": value_10k,
        "value_0k": value_0k,
        "depreciation_10k": depreciation_10k,
        "depreciation_0k": depreciation_0k,
        "reliable": reliable,
    }


def identify_undervalued_cars(
    data: pd.DataFrame,
    config: dict,
    listing_ids: pd.Series,
    predictor: TabularPredictor,
) -> None:
    """Identify undervalued cars using AutoGluon predictions.

    Prints a list of cars where the actual price is below the predicted value
    by more than the configured thresholds.

    Args:
        data: DataFrame with features and price.
        config: Configuration with undervalued_threshold_min/max and final_filters.
        listing_ids: Series of listing IDs corresponding to data rows.
        predictor: Trained AutoGluon TabularPredictor.
    """
    from utils import apply_filters

    predictions = predictor.predict(data)
    data = data.copy()
    data["prediction_label"] = predictions

    undervalued_filter_min = config["undervalued_threshold_min"]
    undervalued_filter_max = config["undervalued_threshold_max"]
    undervalued = data[
        (data["price"] < data["prediction_label"] - undervalued_filter_min)
        & (data["price"] > data["prediction_label"] - undervalued_filter_max)
    ]
    undervalued = undervalued.merge(
        listing_ids.to_frame(), left_index=True, right_index=True
    )

    filters = config.get("final_filters", [])
    filtered_dataframe = apply_filters(undervalued, filters)

    print(f"\nFound {len(filtered_dataframe)} undervalued cars:")
    print(filtered_dataframe)


# =============================================================================
# Interactive Input Helpers
# =============================================================================


def get_input_with_match(prompt: str, options: list, threshold: int = 80) -> str:
    """Get user input and fuzzy match against valid options.

    Args:
        prompt: Prompt text to display.
        options: List of valid options.
        threshold: Minimum fuzzy match score (0-100).

    Returns:
        Best matching option from the list.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            print("  Please enter a value.")
            continue
        best_match = process.extractOne(user_input, options)
        if best_match[1] < threshold:
            print(f"  Not recognised. Valid options: {', '.join(map(str, options))}")
            continue
        if best_match[1] < 100:
            print(f"  -> Matched to: {best_match[0]}")
        return best_match[0]


def get_numeric_input(prompt: str, unit_conversions: dict = None) -> float:
    """Get numeric input with optional unit conversion.

    Args:
        prompt: Prompt text to display.
        unit_conversions: Optional dict mapping unit strings to multipliers,
                          e.g., {"l": 1000, "cc": 1} for engine size.

    Returns:
        Numeric value (with unit conversion applied if specified).
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            print("  Please enter a value.")
            continue
        try:
            value = user_input.lower()
            if unit_conversions:
                for unit, multiplier in unit_conversions.items():
                    if unit in value:
                        value = value.replace(unit, "").strip()
                        return float(value) * multiplier
            return float(value.replace(",", ""))
        except ValueError:
            print("  Please enter a valid number.")
            continue


# =============================================================================
# Valuation Workflows
# =============================================================================


def run_interactive_valuation() -> None:
    """Run an interactive car valuation session.

    Prompts the user for car specifications, performs classification
    and prediction, and displays the valuation result with confidence
    intervals and depreciation forecasts.
    """
    from sentence_transformers import SentenceTransformer
    from feature_engineering import build_data_point, classify_colour, classify_stereo

    config = load_config()
    car_make = config["make"]
    car_model = config["model"]

    print(f"\n{'='*50}")
    print(f"  Car Valuation Tool - {car_make} {car_model}")
    print(f"{'='*50}\n")

    print("Loading models...")
    model_path = get_model_path(car_make, car_model)
    predictor = TabularPredictor.load(model_path)
    embedding_model = SentenceTransformer(config["sentence_transformer"]["model"])
    print("Models loaded.\n")

    regions = [
        "Auckland", "Bay of Plenty", "Canterbury", "Gisborne",
        "Hawke's Bay", "Manawatu", "Marlborough", "Nelson Bays",
        "Northland", "Otago", "Southland", "Taranaki",
        "Timaru - Oamaru", "Waikato", "Wairarapa", "Wellington",
        "West Coast", "Whanganui",
    ]
    fuel_types = ["Petrol", "Diesel", "Electric", "Hybrid"]
    transmissions = ["Automatic", "Manual"]
    cylinders = [4, 6, 8, 10, 12]

    while True:
        print("-" * 50)
        print("Enter car details (or 'q' to quit):\n")

        while True:
            region_input = input("Region: ").strip()
            if region_input.lower() == "q":
                print("\nGoodbye!")
                return
            if not region_input:
                print("  Please enter a region.")
                continue
            best_match = process.extractOne(region_input, regions)
            if best_match[1] < 80:
                print(f"  Not recognised. Valid options: {', '.join(regions)}")
                continue
            region = best_match[0]
            if best_match[1] < 100:
                print(f"  -> Matched to: {region}")
            break

        engine_size = get_numeric_input(
            "Engine size (e.g., 1.2L or 1200cc): ",
            unit_conversions={"l": 1000, "cc": 1},
        )
        odometer = get_numeric_input("Odometer (km): ", unit_conversions={"km": 1})
        year = int(get_numeric_input("Year (e.g., 2005): "))
        fuel_type = get_input_with_match(
            f"Fuel type ({'/'.join(fuel_types)}): ", fuel_types
        )
        transmission = get_input_with_match(
            f"Transmission ({'/'.join(transmissions)}): ", transmissions
        )
        cylinder = int(
            get_input_with_match(
                f"Cylinders ({'/'.join(map(str, cylinders))}): ",
                cylinders, threshold=0,
            )
        )
        exterior_colour = input("Exterior colour: ").strip() or "Unknown"
        stereo_description = input("Stereo description: ").strip() or "Standard"
        is4wd_input = input("4WD? (y/n): ").strip().lower()
        is_4wd = 1 if is4wd_input in ["y", "yes"] else 0

        print("\nProcessing...")
        colour_label = classify_colour(
            pd.DataFrame({"ExteriorColour": [exterior_colour]}), embedding_model,
            car_make, car_model,
        ).iloc[0]["colour_label"]
        stereo_label = classify_stereo(
            pd.DataFrame({"StereoDescription": [stereo_description]}), embedding_model,
            car_make, car_model,
        ).iloc[0]["stereo_label"]

        data_point = build_data_point(
            region, engine_size, odometer, year, fuel_type, transmission,
            cylinder, colour_label, stereo_label, is_4wd,
        )

        predicted_price = predictor.predict(data_point).iloc[0]

        print("\n" + "=" * 50)
        print("  VALUATION RESULT")
        print("=" * 50)
        print(f"\n  {year} {car_make} {car_model}")
        print(f"  {engine_size:.0f}cc, {fuel_type}, {transmission}")
        print(f"  {odometer:,.0f} km, {region}")
        print(f"\n  Estimated Value: ${predicted_price:,.0f}")

        # Confidence intervals if quantile model exists
        result = predict_with_confidence(data_point, car_make, car_model)
        if "ci_80" in result:
            print(f"  80% Confidence:  ${result['ci_80'][0]:,.0f} - ${result['ci_80'][1]:,.0f}")
        if "ci_50" in result:
            print(f"  50% Confidence:  ${result['ci_50'][0]:,.0f} - ${result['ci_50'][1]:,.0f}")
        if result.get("tail"):
            print(f"  (Tail-adjusted: {result['tail']} end)")

        print("=" * 50 + "\n")

        another = input("Value another car? (y/n): ").strip().lower()
        if another not in ["y", "yes"]:
            print("\nGoodbye!")
            break


def value_car(config_file: str = "car_to_value.yml") -> None:
    """Value a car using specs from a config file.

    Loads car specifications from the config file, performs classification
    and prediction, and displays comprehensive valuation results including
    confidence intervals, regional comparison, and depreciation forecast.

    Args:
        config_file: Path to the YAML file containing car specifications.
    """
    from sentence_transformers import SentenceTransformer
    from feature_engineering import build_data_point, classify_colour, classify_stereo
    from utils import load_yaml

    car_specs = load_yaml(config_file)

    car_make = car_specs["make"]
    car_model = car_specs["model"]
    config = load_config()

    print(f"\n{'='*50}")
    print(f"  Car Valuation - {car_make} {car_model}")
    print(f"{'='*50}\n")

    # Load AutoGluon model
    model_path = get_model_path(car_make, car_model)
    if not os.path.isdir(model_path):
        print(f"ERROR: No model found at {model_path}")
        return

    print(f"Loading: {model_path}")
    predictor = TabularPredictor.load(model_path)
    print("Model type: AutoGluon ensemble")

    embedding_model = SentenceTransformer(config["sentence_transformer"]["model"])
    print()

    # Parse specs from config
    region = car_specs["region"]
    year = int(car_specs["year"])

    engine_size_str = str(car_specs["engine_size"]).lower()
    if "l" in engine_size_str:
        engine_size = float(engine_size_str.replace("l", "").strip()) * 1000
    elif "cc" in engine_size_str:
        engine_size = float(engine_size_str.replace("cc", "").strip())
    else:
        engine_size = float(engine_size_str)

    odometer = float(car_specs["odometer"])
    fuel_type = car_specs["fuel_type"]
    transmission = car_specs["transmission"]
    cylinder = int(car_specs["cylinders"])
    exterior_colour = car_specs.get("exterior_colour", "Unknown")
    stereo_description = car_specs.get("stereo_description", "Standard")
    is_4wd = 1 if car_specs.get("is_4wd", False) else 0

    # Classify colour and stereo
    print("Processing...")
    colour_label = classify_colour(
        pd.DataFrame({"ExteriorColour": [exterior_colour]}), embedding_model,
        car_make, car_model,
    ).iloc[0]["colour_label"]
    stereo_label = classify_stereo(
        pd.DataFrame({"StereoDescription": [stereo_description]}), embedding_model,
        car_make, car_model,
    ).iloc[0]["stereo_label"]

    data_point = build_data_point(
        region, engine_size, odometer, year, fuel_type, transmission,
        cylinder, colour_label, stereo_label, is_4wd,
    )

    # Point estimate
    predicted_price = predictor.predict(data_point).iloc[0]

    # Confidence intervals
    result = predict_with_confidence(data_point, car_make, car_model)

    print("\n" + "=" * 50)
    print("  VALUATION RESULT")
    print("=" * 50)
    print(f"\n  {year} {car_make} {car_model}")
    print(f"  {engine_size:.0f}cc, {fuel_type}, {transmission}")
    print(f"  {odometer:,.0f} km, {region}")
    print(f"  {cylinder} cylinders, {'4WD' if is_4wd else '2WD'}")
    print(f"  Colour: {exterior_colour}")
    print(f"\n  Estimated Value: ${result['estimate']:,.0f}")

    if "ci_80" in result:
        print(f"  80% Confidence:  ${result['ci_80'][0]:,.0f} - ${result['ci_80'][1]:,.0f}")
    if "ci_50" in result:
        print(f"  50% Confidence:  ${result['ci_50'][0]:,.0f} - ${result['ci_50'][1]:,.0f}")
    if result.get("tail"):
        print(f"  (Tail-adjusted: {result['tail']} end)")

    print("=" * 50)

    # Regional comparison if specified
    comparison_region = car_specs.get("comparison_region")
    if comparison_region and comparison_region != region:
        comparison_data_point = build_data_point(
            comparison_region, engine_size, odometer, year, fuel_type, transmission,
            cylinder, colour_label, stereo_label, is_4wd,
        )
        comparison_price = predictor.predict(comparison_data_point).iloc[0]
        price_diff = comparison_price - predicted_price
        price_diff_pct = (price_diff / predicted_price) * 100

        print("\n" + "-" * 50)
        print("  REGIONAL COMPARISON")
        print("-" * 50)
        print(f"  Value in {region}:           ${predicted_price:,.0f}")
        print(f"  Value in {comparison_region}:              ${comparison_price:,.0f}")
        if price_diff >= 0:
            print(f"  Difference:                  +${price_diff:,.0f} (+{price_diff_pct:.1f}%)")
        else:
            print(f"  Difference:                  -${-price_diff:,.0f} ({price_diff_pct:.1f}%)")
        print("-" * 50)

    # Calculate and display depreciation
    depreciation = calculate_depreciation(predictor, data_point, predicted_price)

    print("\n" + "-" * 50)
    print("  12-MONTH DEPRECIATION FORECAST")
    print("-" * 50)

    if not depreciation["reliable"]:
        print("  * Estimates may be unreliable for this vehicle *")
        print()

    dep_10k = depreciation["depreciation_10k"]
    dep_0k = depreciation["depreciation_0k"]
    val_10k = depreciation["value_10k"]
    val_0k = depreciation["value_0k"]

    def fmt_dep(dep, val):
        if dep >= 0:
            return f"-${dep:,.0f} ({dep/predicted_price*100:.1f}%)  -> ${val:,.0f}"
        else:
            return f"+${-dep:,.0f} ({-dep/predicted_price*100:.1f}%)  -> ${val:,.0f}"

    print(f"  With 10,000 km travel:  {fmt_dep(dep_10k, val_10k)}")
    print(f"  With minimal travel:    {fmt_dep(dep_0k, val_0k)}")
    print("=" * 50 + "\n")
