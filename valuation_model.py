import pandas as pd
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model, save_model, plot_model
from pycaret.regression import compare_models, load_model
import yaml
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
import sys
from fuzzywuzzy import process


def read_config(filename='config.yml'):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_data(filename='data.pkl'):
    data = pd.read_pickle(filename)
    return data


def classify_colour(data, embedding_model):
    embeds = embedding_model.encode(data['ExteriorColour'].values)
    model = load_model('colour_classifier_model')
    prediction_data = pd.concat([data['ExteriorColour'].reset_index(), pd.DataFrame(embeds)], axis=1)
    prediction_data = prediction_data.drop('index', axis=1)
    predictions = predict_model(model, data=prediction_data)
    data['colour_label'] = predictions['prediction_label']
    return data


def classify_stereo(data, embedding_model):
    embeds = embedding_model.encode(data['StereoDescription'].values)
    model = load_model('stereo_classifier_model')
    prediction_data = pd.concat([data['StereoDescription'].reset_index(), pd.DataFrame(embeds)], axis=1)
    prediction_data = prediction_data.drop('index', axis=1)
    predictions = predict_model(model, data=prediction_data)
    data['stereo_label'] = predictions['prediction_label']
    return data


def feature_engineering(data, config):
    data['price'] = data['PriceDisplay'].str.extract(r'\$([0-9,]+)')
    data['price'] = data['price'].str.replace(',', '').astype(int)
    cols_to_keep = config['feature_engineering']['columns_to_keep']
    cols_to_keep = [col for col in cols_to_keep if col in data.columns]
    data = data[cols_to_keep]
    data = pd.get_dummies(data, columns=['Region'], drop_first=True)
    data = pd.get_dummies(data, columns=['Transmission'], drop_first=True)
    data = pd.get_dummies(data, columns=['Fuel'], drop_first=True)
    data = pd.get_dummies(data, columns=['Cylinders'], drop_first=True)
    model = SentenceTransformer(config['sentence_transformer']['model'])
    data['ExteriorColour'] = data['ExteriorColour'].fillna('No colour')
    data = classify_colour(data, model)
    data = data.drop('ExteriorColour', axis=1)
    data = pd.get_dummies(data, columns=['colour_label'], drop_first=True)
    data['StereoDescription'] = data['StereoDescription'].fillna('No stereo')
    data = classify_stereo(data, model)
    data = data.drop('StereoDescription', axis=1)
    data = pd.get_dummies(data, columns=['stereo_label'], drop_first=True)
    if 'IsNew' in data.columns:
        data['IsNew'] = data['IsNew'].fillna(0)
        data['IsNew'] = data['IsNew'].astype(int)
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
    try:
        data = data[data['BidCount'].isnull()].reset_index()
    except KeyError:
        pass
    data = feature_engineering(data, config)
    data_point_to_valuate = data[data['ListingId'] == config['valuation_listing_id']]
    listingIds = data['ListingId']
    data = data.drop(data_point_to_valuate.index)
    data = data.drop('ListingId', axis=1)
    out_of_sample = data.sample(frac=0.4, random_state=42)
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
    if data_point_to_valuate.shape[0] > 0:
        valuation = predict_model(final_rf, data=data_point_to_valuate)
        print(valuation)
    # plot_model(final_rf, plot='residuals')
    # plot_model(final_rf, plot='error')
    # plot_model(final_rf, plot='feature')
    full_data = pd.concat([data, out_of_sample, data_point_to_valuate])
    identify_undervalued_cars(full_data, config, listingIds, final_rf)


def value_input_car():
    config = read_config()
    car_make = config['make']
    car_model = config['model']
    model = load_model(f'{car_make}_{car_model}_final_rf_model')
    embedding_model = SentenceTransformer(config['sentence_transformer']['model'])
    regions = ['Bay of Plenty',
               'Canterbury',
               'Gisborne',
               "Hawke's Bay",
               'Manawatu',
               'Marlborough',
               'Nelson Bays',
               'Northland',
               'Otago',
               'Southland',
               'Taranaki',
               'Timaru - Oamaru',
               'Waikato',
               'Wairarapa',
               'Wellington',
               'West Coast',
               'Whanganui']
    while True:
        region = input('Enter the region: ')
        # Match the input to one of the regions
        best_match = process.extractOne(region, regions)
        if best_match[1] < 80:
            print('Error: region not recognised')
            sys.exit(1)
        else:
            region = best_match[0]
            print(f'Best match: {region}')
        engine_size = input('Enter the engine size: ')
        if 'L' in engine_size:
            engine_size = engine_size.replace('L', '')
            engine_size = float(engine_size) * 1000
        elif 'cc' in engine_size:
            engine_size = engine_size.replace('cc', '')
            engine_size = float(engine_size)
        odometer = input('Enter the odometer reading: ')
        if 'km' in odometer:
            odometer = odometer.replace('km', '')
            odometer = float(odometer)
        year = input('Enter the year: ')
        year = int(year)
        fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
        fuel_type = input('Enter the fuel type: ')
        similarities = process.extractOne(fuel_type, fuel_types)
        if similarities[1] < 80:
            print('Error: fuel type not recognised')
            sys.exit(1)
        else:
            fuel_type = similarities[0]
            print(f'Best match: {fuel_type}')
        transmissions = ['Automatic', 'Manual']
        transmission = input('Enter the transmission type: ')
        similarities = process.extractOne(transmission, transmissions)
        if similarities[1] < 80:
            print('Error: transmission type not recognised')
            sys.exit(1)
        else:
            transmission = similarities[0]
            print(f'Best match: {transmission}')
        cylinders = [4, 6, 8, 10, 12]
        cylinder = input('Enter the number of cylinders: ')
        cylinder = int(cylinder)
        differences = [abs(cylinder - c) for c in cylinders]
        cylinder = cylinders[differences.index(min(differences))]
        print(f'Best match: {cylinder}')
        exterior_colour = input('Enter the exterior colour: ')
        stereo_description = input('Enter the stereo description: ')
        is4Wd = input('Is the car 4WD? (y/n): ')
        if is4Wd == 'y' or is4Wd == 'Y' or is4Wd == 'yes' or is4Wd == 'Yes':
            is4Wd = 1
        else:
            is4Wd = 0
        isNew = False
        isDealer = False
        colour_label = classify_colour(pd.DataFrame({'ExteriorColour': [exterior_colour]}), embedding_model).iloc[0]['colour_label']
        stereo_label = classify_stereo(pd.DataFrame({'StereoDescription': [stereo_description]}), embedding_model).iloc[0]['stereo_label']
        data_point = pd.DataFrame({f'Region_{region}': 1,
                                'EngineSize': [engine_size],
                                'Odometer': [odometer],
                                'Year': [year - 2000],
                                f'Transmission_{transmission}': 1,
                                f'Fuel_{fuel_type}': 1,
                                f'Cylinders_{cylinder}': 1,
                                f'ExteriorColour_cluster_{colour_label}': 1,
                                f'StereoDescription_cluster_{stereo_label}': 1,
                                'Is4WD': [is4Wd],
                                'IsNew': [isNew],
                                'IsDealer': [isDealer]})
        columns = model.feature_names_in_
        missing_columns = list(set(columns) - set(data_point.columns))
        for column in missing_columns:
            data_point[column] = 0
        data_point = data_point[columns]
        valuation = predict_model(model, data=data_point)
        print(valuation)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        value_input_car()
    else:
        print('Error: too many input parameters')
        sys.exit(1)
