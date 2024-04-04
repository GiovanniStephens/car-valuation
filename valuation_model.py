import pandas as pd
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model, save_model, plot_model
from pycaret.regression import compare_models,load_model
import yaml
from yaml.loader import SafeLoader


def read_config(filename='config.yml'):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_data(filename='data.pkl'):
    data = pd.read_pickle(filename)
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
    data = pd.get_dummies(data, columns=['ExteriorColour'], drop_first=True)
    data = pd.get_dummies(data, columns=['StereoDescription'], drop_first=True)
    data['IsNew'] = data['IsNew'].fillna(0)
    data['Year'] = data['Year'] - 2000
    return data


def main():
    config = read_config()
    make = config['make']
    model = config['model']
    data = load_data(f'{make}_{model}_data.pkl')
    data = feature_engineering(data, config)
    data_point_to_valuate = data[data['ListingId'] == 4627127218]
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
    best_model = compare_models(sort='MAE')
    rf = create_model('rf')
    tuned_rf = tune_model(rf)
    final_rf = finalize_model(tuned_rf)
    save_model(final_rf, 'final_rf_model')
    predictions = predict_model(final_rf, data=out_of_sample)
    valuation = predict_model(final_rf, data=data_point_to_valuate)
    print(valuation)
    plot_model(final_rf, plot='residuals')
    # plot_model(final_rf, plot='error')
    # plot_model(final_rf, plot='feature')


if __name__ == '__main__':
    main()