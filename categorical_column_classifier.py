import yaml
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
import hdbscan
import pandas as pd
from pycaret.classification import (
    setup,
    create_model,
    tune_model,
    finalize_model,
    predict_model,
    save_model,
    plot_model,
)


def read_config(filename="data/config.yml"):
    with open(filename, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_data(filename="data/data.pkl"):
    data = pd.read_pickle(filename)
    return data


def main(column_name="ExteriorColour"):
    config = read_config()
    make = config["make"]
    model = config["model"]
    data = load_data(f"data/{make}_{model}_data.pkl")
    model = SentenceTransformer(config["sentence_transformer"]["model"])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
    if column_name == "ExteriorColour":
        data["ExteriorColour"] = data["ExteriorColour"].fillna("No colour")
        embeds = model.encode(data["ExteriorColour"].values)
    elif column_name == "StereoDescription":
        data["StereoDescription"] = data["StereoDescription"].fillna("No stereo")
        embeds = model.encode(data["StereoDescription"].values)
    clusterer.fit(embeds)
    data["cluster"] = clusterer.labels_
    cluster_labels = data.groupby("cluster")[column_name].agg(
        lambda x: x.value_counts().index[0]
    )
    data["cluster"] = data["cluster"].map(cluster_labels)
    training_data = pd.concat(
        [data["cluster"].reset_index(), pd.DataFrame(embeds)], axis=1
    )
    training_data = training_data.drop("index", axis=1)
    _ = setup(training_data, target="cluster")
    rf = create_model("rf")
    tuned_rf = tune_model(rf)
    final_rf = finalize_model(tuned_rf)
    _ = predict_model(final_rf, data=training_data)
    if column_name == "ExteriorColour":
        save_model(final_rf, "models/colour_classifier_model")
    elif column_name == "StereoDescription":
        save_model(final_rf, "models/stereo_classifier_model")
    plot_model(final_rf, plot="confusion_matrix")


if __name__ == "__main__":
    main("StereoDescription")
