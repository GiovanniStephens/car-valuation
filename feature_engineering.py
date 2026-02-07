"""
Feature engineering module for car valuation.

This module handles transforming raw car listing data into features suitable
for AutoGluon model training and prediction.
"""

import os

import pandas as pd
from sentence_transformers import SentenceTransformer

from model_cache import get_predictor

CLASSIFIER_PATHS = {
    "colour": "models/colour_classifier",
    "stereo": "models/stereo_classifier",
}


def classify_colour(
    data: pd.DataFrame, embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """Classify exterior colour descriptions into standardized labels.

    Uses a pre-trained AutoGluon classifier with sentence embeddings to map
    free-text colour descriptions to consistent labels (e.g., "White", "Black", "Silver").
    Falls back to "Unknown" if the classifier is not available.

    Args:
        data: DataFrame with an "ExteriorColour" column containing colour descriptions.
        embedding_model: Sentence transformer model for encoding text.

    Returns:
        DataFrame with a new "colour_label" column containing classified colours.
    """
    data = data.copy()
    classifier_path = CLASSIFIER_PATHS["colour"]
    if not os.path.isdir(classifier_path):
        data["colour_label"] = "Unknown"
        return data

    embeds = embedding_model.encode(data["ExteriorColour"].values)
    predictor = get_predictor(classifier_path)
    prediction_data = pd.DataFrame(embeds)
    data["colour_label"] = predictor.predict(prediction_data).values
    return data


def classify_stereo(
    data: pd.DataFrame, embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """Classify stereo descriptions into standardized labels.

    Uses a pre-trained AutoGluon classifier with sentence embeddings to map
    free-text stereo descriptions to consistent labels (e.g., "Basic", "Premium").
    Falls back to "Unknown" if the classifier is not available.

    Args:
        data: DataFrame with a "StereoDescription" column.
        embedding_model: Sentence transformer model for encoding text.

    Returns:
        DataFrame with a new "stereo_label" column containing classified stereo types.
    """
    data = data.copy()
    classifier_path = CLASSIFIER_PATHS["stereo"]
    if not os.path.isdir(classifier_path):
        data["stereo_label"] = "Unknown"
        return data

    embeds = embedding_model.encode(data["StereoDescription"].values)
    predictor = get_predictor(classifier_path)
    prediction_data = pd.DataFrame(embeds)
    data["stereo_label"] = predictor.predict(prediction_data).values
    return data


def feature_engineering(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare features for AutoGluon model training or prediction.

    This function transforms raw car listing data into a feature set suitable
    for AutoGluon. Key transformations:
    - Extracts numeric price from PriceDisplay string
    - Classifies colour and stereo descriptions using pre-trained models
    - Computes derived features (Age, KmPerYear)
    - Ensures categorical columns are strings (AutoGluon handles them natively)

    Args:
        data: DataFrame containing raw car listing data with columns like
              PriceDisplay, ExteriorColour, StereoDescription, Year, Odometer, etc.
        config: Configuration dictionary with keys:
                - sentence_transformer.model: Name of the embedding model
                - feature_engineering.columns_to_keep: List of columns to retain

    Returns:
        DataFrame with engineered features ready for model training/prediction.
    """
    data = data.copy()
    data["price"] = data["PriceDisplay"].str.extract(r"\$([0-9,]+)")
    data["price"] = data["price"].str.replace(",", "").astype(int)

    cols_to_keep = config["feature_engineering"]["columns_to_keep"]
    cols_to_keep = [col for col in cols_to_keep if col in data.columns]
    data = data[cols_to_keep]

    # Colour and stereo classification -> string labels (no one-hot)
    embedding_model = SentenceTransformer(config["sentence_transformer"]["model"])

    data["ExteriorColour"] = data["ExteriorColour"].fillna("No colour")
    data = classify_colour(data, embedding_model)
    data = data.drop("ExteriorColour", axis=1)

    data["StereoDescription"] = data["StereoDescription"].fillna("No stereo")
    data = classify_stereo(data, embedding_model)
    data = data.drop("StereoDescription", axis=1)

    # Derived features
    data["Age"] = 2026 - data["Year"]
    data["KmPerYear"] = data["Odometer"] / data["Age"].clip(lower=1)

    # Boolean/int columns
    if "IsNew" in data.columns:
        data["IsNew"] = data["IsNew"].fillna(0).astype(int)

    # Categorical columns stay as strings -- AutoGluon handles them natively
    for col in [
        "Region",
        "Transmission",
        "Fuel",
        "Cylinders",
        "colour_label",
        "stereo_label",
    ]:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data


def build_data_point(
    region: str,
    engine_size: float,
    odometer: float,
    year: int,
    fuel_type: str,
    transmission: str,
    cylinder: int,
    colour_label: str,
    stereo_label: str,
    is_4wd: int,
    is_new: bool = False,
    is_dealer: bool = False,
) -> pd.DataFrame:
    """Build a single-row DataFrame for prediction with AutoGluon.

    Creates a properly formatted data point from individual car specifications.
    No one-hot encoding is needed as AutoGluon handles categorical columns natively.

    Args:
        region: Geographic region (e.g., "Auckland", "Canterbury").
        engine_size: Engine displacement in cc.
        odometer: Odometer reading in km.
        year: Model year (e.g., 2018).
        fuel_type: Fuel type ("Petrol", "Diesel", "Electric", "Hybrid").
        transmission: Transmission type ("Automatic", "Manual").
        cylinder: Number of cylinders (4, 6, 8, etc.).
        colour_label: Classified colour label.
        stereo_label: Classified stereo label.
        is_4wd: 1 if 4WD, 0 otherwise.
        is_new: True if the car is new, False otherwise.
        is_dealer: True if sold by dealer, False otherwise.

    Returns:
        Single-row DataFrame with all required feature columns.
    """
    age = 2026 - year
    km_per_year = odometer / max(age, 1)

    return pd.DataFrame(
        {
            "Region": [region],
            "EngineSize": [engine_size],
            "Odometer": [odometer],
            "Year": [year],
            "Transmission": [transmission],
            "Fuel": [fuel_type],
            "Cylinders": [str(cylinder)],
            "colour_label": [colour_label],
            "stereo_label": [stereo_label],
            "Is4WD": [is_4wd],
            "IsNew": [int(is_new)],
            "IsDealer": [int(is_dealer)],
            "Age": [age],
            "KmPerYear": [km_per_year],
        }
    )
