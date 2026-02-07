import glob
import os
import sys

import yaml
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
import hdbscan
import pandas as pd
from autogluon.tabular import TabularPredictor

FILL_VALUES = {
    "ExteriorColour": "No colour",
    "StereoDescription": "No stereo",
}

SAVE_PATHS = {
    "ExteriorColour": "models/colour_classifier",
    "StereoDescription": "models/stereo_classifier",
}


def load_combined_column(column_name):
    """Load and combine a single column from all data files."""
    data_files = glob.glob("data/*_data.pkl")
    if not data_files:
        print("ERROR: No data files found in data/")
        sys.exit(1)

    all_values = []
    for data_file in data_files:
        basename = os.path.basename(data_file)
        try:
            data = pd.read_pickle(data_file)
            if column_name not in data.columns:
                print(f"  Skipping {basename} (no {column_name} column)")
                continue
            values = data[column_name].fillna(FILL_VALUES[column_name])
            all_values.append(values)
            print(f"  Loaded {len(values)} rows from {basename}")
        except Exception as e:
            print(f"  Error loading {basename}: {e}")
            continue

    if not all_values:
        print(f"ERROR: No data found for column {column_name}")
        sys.exit(1)

    combined = pd.concat(all_values, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(all_values)} files")
    return combined


def main(column_name="ExteriorColour"):
    with open("model_settings.yml", "r") as f:
        model_config = yaml.load(f, Loader=SafeLoader)

    combined = load_combined_column(column_name)

    # Skip training if all values are the fill value (no real data)
    real_values = combined[combined != FILL_VALUES[column_name]]
    if len(real_values) == 0:
        print(f"  No real data for {column_name} — skipping classifier training.")
        return

    embedding_model = SentenceTransformer(model_config["text_embedding_model"])
    embeds = embedding_model.encode(combined.values)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
    clusterer.fit(embeds)
    labels = pd.Series(clusterer.labels_)

    # Filter out noise points (label -1) before building training labels
    noise_mask = labels == -1
    n_noise = noise_mask.sum()
    if n_noise > 0:
        print(f"  Dropping {n_noise} noise points from {len(combined)} total")
    combined = combined[~noise_mask].reset_index(drop=True)
    labels = labels[~noise_mask].reset_index(drop=True)
    embeds = embeds[~noise_mask.values]

    # Map cluster IDs to the most common original value in each cluster
    cluster_labels = pd.DataFrame({"cluster": labels, "value": combined})
    label_map = cluster_labels.groupby("cluster")["value"].agg(
        lambda x: x.value_counts().index[0]
    )
    target = labels.map(label_map)

    training_data = pd.concat(
        [pd.DataFrame(embeds), target.rename("cluster").reset_index(drop=True)], axis=1
    )

    save_path = SAVE_PATHS[column_name]
    predictor = TabularPredictor(
        label="cluster",
        problem_type="multiclass",
        path=save_path,
    )
    predictor.fit(training_data, time_limit=120)
    print(f"  Saved classifier to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
    else:
        columns = ["ExteriorColour", "StereoDescription"]

    for col in columns:
        print(f"\nTraining {col} classifier...")
        main(col)
    print("\nDone.")
