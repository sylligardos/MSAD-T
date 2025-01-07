"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

from data.metricloader import Metricloader
from data.dataloader import Dataloader
from utils.split_ts import my_train_test_split

import argparse
import numpy as np
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from aeon.regression.deep_learning import FCNRegressor, InceptionTimeRegressor, ResNetRegressor
import pickle
import sklearn


def create_directory(parent_dir, new_dir_name):
    new_dir_path = os.path.join(parent_dir, new_dir_name)
    
    # Check if the directory exists; if not, create it
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
        print(f"Directory created: {new_dir_path}")
    
    return new_dir_path

def get_regressor(index, model_class):
    aeon_regressors = [
        ('FCNRegressor', FCNRegressor),
        ('InceptionTimeRegressor', InceptionTimeRegressor),
        ('ResNetRegressor', ResNetRegressor),
    ]

    sklearn_regressors = [
        ('RandomForestRegressor', sklearn.ensemble.RandomForestRegressor()),
        ('SupportVectorRegression', sklearn.svm.SVR()),
        ('XGBRegressor', xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, verbosity=1))
        # ('Ridge', sklearn.linear_model.Ridge())
    ]

    if model_class == 'raw':
        curr_regressor_name, curr_regressor = aeon_regressors[index]
        model_parameters = {'verbose':True, 'save_best_model':True, 'best_file_name':'best_model', 'file_path':''} 
    elif model_class == 'feature':
        curr_regressor_name, curr_regressor = sklearn_regressors[index]
        model_parameters = None
    
    return curr_regressor_name, curr_regressor, model_parameters

def log_time(msg, start_time, curr_time):
    elapsed_time = time.time() - start_time
    current_elapsed_time = time.time() - curr_time
    full_msg = f"[et: {elapsed_time:.2f} s, cet: {current_elapsed_time:.2f} s]: {msg}"
    
    print(f">> {full_msg}")

    return time.time()

def train_regressors(
        model_index,
        model_class,
        detector,
        window_size,
        experiment,
        split,
        saving_path,
        testing=False,
    ):
    # Experiment setup
    curr_time = tic = time.time()
    model_name, model, model_parameters = get_regressor(int(model_index), model_class)

    # Load the scores of all detectors
    # scoreloader = Scoreloader('data/scores')
    # detectors = scoreloader.get_detector_names()

    # Create subdirs (if they dont exist) and essential paths
    saving_path_split = os.path.split(saving_path)
    create_directory(saving_path_split[0], saving_path_split[1])
    experiment_name = f"{model_name}_{model_class}_{window_size}_{detector}_{experiment}"
    if experiment == 'unsupervised':
        experiment_name += f"_{split}"
    model_saving_path = os.path.join(create_directory(saving_path, 'models'), experiment_name + '.json')
    log_file = os.path.join(create_directory(saving_path, 'logs'), experiment_name + '.log')
    results_path = os.path.join(create_directory(saving_path, 'results'), experiment_name + '.csv')
    training_info_path = os.path.join(create_directory(saving_path, 'training_info'), experiment_name + '.csv')
    
    # Setup logger
    curr_time = log_time(f'(Starting experiment) Model: {model_name}, Window size: {window_size}, Detector: {detector}, Experiment: {experiment}, Split: {split}', tic, curr_time)

    # Load the data features or the raw subsequences 
    datasets = ['YAHOO'] #if testing else None
    curr_time = log_time(f"Loading data", tic, curr_time)
    split_file = f'data/splits/{experiment}/' + ('split_TSB' if split is None else f'unsupervised_testsize_1_split_{split}') + '.csv'
    dataloader = Dataloader(
        raw_data_path='data/raw', 
        split_file=split_file,
        feature_data_path=f'data/features/CATCH22_TSB_{window_size}.csv',
        window_data_path=f'data/TSB_{window_size}',
    )
    if model_class.lower() == 'raw':
        data = dataloader.load_all_window_timeseries_parallel() if datasets is None else dataloader.load_multiple_window_timeseries_parallel(datasets)
    elif model_class.lower() == 'feature':
        data = dataloader.load_feature_all() if datasets is None else dataloader.load_feature_datasets(datasets)
    else:
        raise ValueError(f"Didn't recognize model class: {model_class}")
    data[data.isnull()] = 0

    # Load labels for current detector
    metric = 'VUS_PR'
    detector = detector.upper()
    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric)
    labels_df = metrics_df[detector]

    # Get the different subsets according to the split file
    curr_time = log_time("Splitting the data into subsets", tic, curr_time)
    df_train, df_test = my_train_test_split(data.copy(), stratify=False, split_file=split_file)
    
    # Fix the labels for the current detector
    for df in [df_train, df_test]:
        df['label'] = np.array([labels_df.loc[x] for x in (df.index if window_size == 'entire' else ['.'.join(id.rsplit('.', maxsplit=1)[:-1]) for id in df.index])])
    y_train, x_train = df_train['label'].values, df_train.drop('label', axis=1).values
    y_test, x_test = df_test['label'].values, df_test.drop('label', axis=1).values

    # Train the regressor and predict the test data
    toc = time.time()
    if model_parameters:
        model_parameters['best_file_name'] = os.path.split(model_saving_path)[-1]
        model_parameters['file_path'] = os.path.split(model_saving_path)[:-1][0]
        model = model(**model_parameters)
    model.fit(x_train, y_train)
    training_time = time.time() - toc

    # Predict test data
    toc = time.time()
    y_pred = model.predict(x_test)
    inference_time = time.time() - toc
    y_pred = np.clip(y_pred, 0, 1)

    # Evaluate the trained model on test data
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics for each method
    curr_time = log_time(f"Evaluation Metrics: MAE = {mae:.3f}, MSE: {mse:.3f}, R^2: {r2:.3f}", tic, curr_time)
    
    # Save the trained model (e.g. convnet_aeon_128_NORMA_supervised)
    with open(model_saving_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the results for the test data (save everything, every window prediction)
    results_df = pd.DataFrame(
        data={'label': y_test, 'y_pred': y_pred},
        index=df_test.index
    )
    results_df.to_csv(results_path, index=True)

    # Save training information (1. Model, detector, experiment, etc... 2. Training time, inference time)
    metrics_df = pd.DataFrame({
        'model': model_name,
        'model_class': model_class,
        'window_size': window_size,
        'detector': detector,
        'experiment': experiment,
        'split': split, 
        'training_time': [training_time],
        'inference_time': [inference_time],
        'train_size': [x_train.shape],
        'test_size': [x_test.shape],
        'MAE': mae,
        'MSE': mse,
        'R^2': r2,
    })

    # Save metrics to a CSV file
    metrics_df.to_csv(training_info_path, index=False)

    # How to load model
    # with open(model_saving_path, 'rb') as f:
    #     clf2 = pickle.load(f)
    # clf2.predict(x_test)

    # TODO: Check all experiment argument values before main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_regressors',
        description='Train any available regression model'
    )
    parser.add_argument('-m', '--model_index', type=str, help='index of the model architecture to train (from 0 to 28)', required=True)
    parser.add_argument('-c', '--model_class', type=str, help='belongs to either aeon, sklearn or xgboost regression models', required=True)
    parser.add_argument('-d', '--detector', type=str, help='The detector\'s values you want to predict', required=True)
    parser.add_argument('-w', '--window_size', type=str, help='The size of the input subsequences', required=True)
    parser.add_argument('-e', '--experiment', type=str, help='Type of experiment (supervised or unsupervised)', required=True)
    parser.add_argument('-s', '--split', type=str, help='Which split to use, only for the unsupervised experiments', required=False, default=None)
    parser.add_argument('-p', '--saving_path', type=str, help='where to save the results of the experiment', required=True)
    parser.add_argument('-t', '--testing', type=bool, help='Testing or normal mode', required=False, default=False)
    
    args = parser.parse_args()

    train_regressors(
        model_index=args.model_index,
        model_class=args.model_class,
        detector=args.detector,
        window_size=args.window_size,
        experiment=args.experiment,
        split=args.split,
        saving_path=args.saving_path,
        testing=args.testing,
    )



""" aeon_regressors = [
    # 'CanonicalIntervalForestRegressor', 
    'FCNRegressor', 
    # 'FreshPRINCERegressor', 
    'InceptionTimeRegressor', 
    # 'KNeighborsTimeSeriesRegressor', 
    # 'MLPRegressor', 
    # 'RandomIntervalRegressor', 
    'ResNetRegressor', 
    # 'RocketRegressor', 
    # 'TimeCNNRegressor', 
    # 'TimeSeriesForestRegressor'
] """