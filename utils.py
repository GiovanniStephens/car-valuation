"""
Shared utilities for the car valuation project.

This module consolidates common functionality used across multiple scripts:
- YAML config loading
- TradeMe API data fetching
- Model metadata I/O
- Filter application
- Model path helpers
"""

import os
import datetime
import yaml
from yaml.loader import SafeLoader
import pandas as pd


# =============================================================================
# Config Loading
# =============================================================================


def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML file contents.
    """
    with open(path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def load_config() -> dict:
    """Load and merge car_search.yml and model_settings.yml into a single config.

    Returns:
        Dictionary containing merged configuration with keys:
        - make: Car make from car_search.yml
        - model: Car model from car_search.yml
        - valuation_listing_id: Optional listing ID to valuate
        - undervalued_threshold_min/max: Discount thresholds
        - final_filters: List of filter rules
        - sentence_transformer: Text embedding model config
        - feature_engineering: Feature columns config
    """
    with open("car_search.yml", "r") as f:
        search_config = yaml.load(f, Loader=SafeLoader)
    with open("model_settings.yml", "r") as f:
        model_config = yaml.load(f, Loader=SafeLoader)

    config = {
        "make": search_config["make"],
        "model": search_config["model"],
        "valuation_listing_id": search_config.get("listing_id_to_valuate"),
        "undervalued_threshold_min": search_config["minimum_discount"],
        "undervalued_threshold_max": search_config["maximum_discount"],
        "final_filters": search_config.get("filters", []),
        "sentence_transformer": {"model": model_config["text_embedding_model"]},
        "feature_engineering": {"columns_to_keep": model_config["features"]},
    }
    return config


def load_training_config() -> dict:
    """Load training configuration from model_settings.yml.

    Returns:
        Dictionary containing training settings (preset, time_limit, etc.)
    """
    with open("model_settings.yml", "r") as f:
        model_config = yaml.load(f, Loader=SafeLoader)
    return model_config.get("training", {})


# =============================================================================
# TradeMe Data Fetching
# =============================================================================


def fetch_trademe_data(make: str, model: str, save_path: str = None) -> pd.DataFrame:
    """Fetch car listing data from the TradeMe API.

    Requires TradeMe API credentials stored in the system keyring under
    the service name "Trademe" with keys "key" and "secret".

    Args:
        make: Car make (e.g., "Toyota").
        model: Car model (e.g., "Corolla").
        save_path: Optional path to save the data as a pickle file.
                   If None, defaults to "data/{make}_{model}_data.pkl".

    Returns:
        DataFrame containing all fetched listings.

    Raises:
        SystemExit: If credentials are missing or no listings are found.
    """
    from requests_oauthlib import OAuth1Session
    import keyring
    import json
    import time
    import sys

    consumer_key = keyring.get_password("Trademe", "key")
    consumer_secret = keyring.get_password("Trademe", "secret")

    if not consumer_key or not consumer_secret:
        print("Error: TradeMe API credentials not found in keyring.")
        print("Please set them using:")
        print('  keyring.set_password("Trademe", "key", "YOUR_KEY")')
        print('  keyring.set_password("Trademe", "secret", "YOUR_SECRET")')
        sys.exit(1)

    trademe_session = OAuth1Session(consumer_key, consumer_secret)

    base_url = f"https://api.trademe.co.nz/v1/Search/Motors/Used.json?make={make}&model={model}&rows=500"

    response = trademe_session.get(base_url)
    parsed_data = json.loads(response.content)
    total_count = parsed_data["TotalCount"]

    if total_count == 0:
        print(f"Error: No listings found for {make} {model}")
        sys.exit(1)

    print(f"Found {total_count} listings")

    n_pages = int(total_count / 500) + 1
    all_data = None

    for i in range(1, n_pages + 1):
        url = base_url + f"&page={i}"
        response = trademe_session.get(url)
        parsed_data = json.loads(response.content)
        listings = parsed_data["List"]
        page_df = pd.DataFrame.from_dict(listings)

        if all_data is None:
            all_data = page_df
        else:
            all_data = pd.concat([all_data, page_df], ignore_index=True)

        print(f"  Page {i}/{n_pages} fetched")
        time.sleep(0.5)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    if save_path is None:
        save_path = f"data/{make}_{model}_data.pkl"

    all_data.to_pickle(save_path)
    print(f"Data saved to {save_path} ({len(all_data)} listings)")

    return all_data


# =============================================================================
# Model Metadata I/O
# =============================================================================


def save_model_metadata(
    model_path: str, n_samples: int, make: str = None, model_name: str = None
) -> None:
    """Save metadata alongside an AutoGluon model.

    Args:
        model_path: Path to the AutoGluon model directory.
        n_samples: Number of training samples.
        make: Optional car make.
        model_name: Optional car model name.
    """
    metadata = {
        "n_training_samples": n_samples,
        "training_date": datetime.datetime.now().isoformat(),
    }
    if make:
        metadata["make"] = make
    if model_name:
        metadata["model"] = model_name
    metadata_path = os.path.join(model_path, "metadata.yml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def load_model_metadata(model_path: str) -> dict:
    """Load metadata for an AutoGluon model.

    Args:
        model_path: Path to the AutoGluon model directory.

    Returns:
        Dictionary containing metadata, or empty dict if no metadata file exists.
    """
    metadata_path = os.path.join(model_path, "metadata.yml")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return yaml.load(f, Loader=SafeLoader)
    return {}


# =============================================================================
# Filter Application
# =============================================================================


def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """Apply a list of filter rules to a DataFrame.

    Each filter rule should be a dict with keys:
    - column: Column name to filter on
    - operator: Comparison operator (==, !=, <, >, <=, >=)
    - value: Value to compare against
    - name: Optional display name for the filter

    Args:
        df: DataFrame to filter.
        filters: List of filter rule dictionaries.

    Returns:
        Filtered DataFrame.
    """
    filtered_df = df.copy()
    for filter_rule in filters or []:
        if filter_rule is None:
            continue
        column = filter_rule.get("column")
        operator = filter_rule.get("operator")
        value = filter_rule.get("value")
        if column and operator and value is not None:
            if column in filtered_df.columns:
                filter_name = filter_rule.get("name", column)
                print(f"Applying filter: {filter_name}")
                filtered_df = filtered_df.query(f"{column} {operator} @value")
    return filtered_df


# =============================================================================
# Model Path Helpers
# =============================================================================


def get_model_path(make: str, model: str) -> str:
    """Get the path to an AutoGluon model directory.

    Args:
        make: Car make.
        model: Car model.

    Returns:
        Path string like "models/Make_Model".
    """
    return f"models/{make}_{model}"


def get_quantile_model_path(make: str, model: str) -> str:
    """Get the path to a quantile model directory.

    Args:
        make: Car make.
        model: Car model.

    Returns:
        Path string like "models/Make_Model_quantile".
    """
    return f"models/{make}_{model}_quantile"


def get_data_path(make: str, model: str) -> str:
    """Get the path to a data file.

    Args:
        make: Car make.
        model: Car model.

    Returns:
        Path string like "data/Make_Model_data.pkl".
    """
    return f"data/{make}_{model}_data.pkl"


def load_data(filename: str = "data/data.pkl") -> pd.DataFrame:
    """Load pickled data from a file.

    Args:
        filename: Path to the pickle file.

    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_pickle(filename)
