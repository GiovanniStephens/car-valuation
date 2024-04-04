import pandas as pd
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model, save_model, plot_model
from pycaret.regression import compare_models, load_model
import yaml
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
import hdbscan
import umap


def read_config(filename='config.yml'):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_data(filename='data.pkl'):
    data = pd.read_pickle(filename)
    return data


def cluster_embeddings(data, model, config, column):
    embeds = model.encode(data[column].values)
    reduced = umap.UMAP(n_components=30).fit_transform(embeds)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    clusterer.fit(reduced)
    data['cluster'] = clusterer.labels_
    cluster_labels = data.groupby('cluster')[column].agg(lambda x: x.value_counts().index[0])
    data['cluster'] = data['cluster'].map(cluster_labels)
    data = pd.get_dummies(data, columns=['cluster'], drop_first=True, prefix=column+'_cluster')
    data = data.drop(column, axis=1)
    return data


def feature_engineering(data, config):
    data['price'] = data['PriceDisplay'].str.extract(r'\$([0-9,]+)')
    data['price'] = data['price'].str.replace(',', '').astype(int)
    cols_to_keep = config['feature_engineering']['columns_to_keep']
    data = data[cols_to_keep]
    data = pd.get_dummies(data, columns=['Region'], drop_first=True)
    data = pd.get_dummies(data, columns=['Transmission'], drop_first=True)
    data = pd.get_dummies(data, columns=['Fuel'], drop_first=True)
    data = pd.get_dummies(data, columns=['Cylinders'], drop_first=True)
    model = SentenceTransformer(config['sentence_transformer']['model'])
    data = cluster_embeddings(data, model, config, 'ExteriorColour')
    data['StereoDescription'] = data['StereoDescription'].fillna('No stereo')
    data = cluster_embeddings(data, model, config, 'StereoDescription')
    data['IsNew'] = data['IsNew'].fillna(0)
    data['Year'] = data['Year'] - 2000
    return data


def identify_undervalued_cars(data, config, listingIds, model):
    predictions = predict_model(model, data=data)
    undervalued_filter_min = config['undervalued_threshold_min']
    undervalued_filter_max = config['undervalued_threshold_max']
    undervalued = predictions[(predictions['price'] < predictions['prediction_label'] - undervalued_filter_min) &
                              (predictions['price'] > predictions['prediction_label'] - undervalued_filter_max)]
    undervalued = undervalued.merge(listingIds, left_index=True, right_index=True)
    filtered_dataframe = undervalued
    filters = config['final_filters']
    for filter_rule in filters:
        column = filter_rule['column']
        operator = filter_rule['operator']
        value = filter_rule['value']
        filtered_dataframe = filtered_dataframe.query(f"{column} {operator} @value")
    print(filtered_dataframe)


def main():
    config = read_config()
    make = config['make']
    model = config['model']
    data = load_data(f'{make}_{model}_data.pkl')
    data = data[data['BidCount'].isnull()]
    data = feature_engineering(data, config)
    data_point_to_valuate = data[data['ListingId'] == config['valuation_listing_id']]
    listingIds = data['ListingId']
    data = data.drop(data_point_to_valuate.index)
    data = data.drop('ListingId', axis=1)

    out_of_sample = data.sample(frac=0.2, random_state=42)
    data = data.drop(out_of_sample.index)
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
    # best_model = compare_models(sort='MAE')
    rf = create_model('rf')
    tuned_rf = tune_model(rf)
    final_rf = finalize_model(tuned_rf)
    save_model(final_rf, f'{make}_{model}_final_rf_model')
    predictions = predict_model(final_rf, data=out_of_sample)
    valuation = predict_model(final_rf, data=data_point_to_valuate)
    print(valuation)
    # plot_model(final_rf, plot='residuals')
    # plot_model(final_rf, plot='error')
    # plot_model(final_rf, plot='feature')
    full_data = pd.concat([data, out_of_sample, data_point_to_valuate])
    identify_undervalued_cars(full_data, config, listingIds, final_rf)


if __name__ == '__main__':
    main()
