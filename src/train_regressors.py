"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

from data.dataloader import Dataloader
from data.metricloader import Metricloader
from data.scoreloader import Scoreloader
from utils.split_ts import my_train_test_split
from utils.utils import create_directory, log_time, get_regressor

import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pandas as pd
import os
import pickle


def train_regressors(
        model_index,
        detector,
        window_size,
        experiment,
        split,
        saving_path,
        testing=False,
    ):
    # Experiment setup
    curr_time = tic = time.time()
    model_name, model, model_class = get_regressor(int(model_index))

    experiment_name = f"{model_name}_{model_class}_{window_size}_{detector}_{experiment}"
    if experiment == 'unsupervised':
        experiment_name += f"_{split}"

    create_directory(*os.path.split(saving_path))
    model_saving_path = os.path.join(create_directory(saving_path, 'models'), experiment_name + '.json')
    log_file = os.path.join(create_directory(saving_path, 'logs'), experiment_name + '.log')
    results_path = os.path.join(create_directory(saving_path, 'results'), experiment_name + '.csv')
    training_info_path = os.path.join(create_directory(saving_path, 'training_info'), experiment_name + '.csv')
    
    curr_time = log_time(f'(Starting experiment) Model: {model_name}, Window size: {window_size}, Detector: {detector}, Experiment: {experiment}, Split: {split}', tic, curr_time)

    # Load data 
    curr_time = log_time(f"Loading data", tic, curr_time)
    datasets = ['YAHOO'] if testing else None
    split_file = f'data/splits/{experiment}/' + ('split_TSB' if split is None else f'unsupervised_testsize_1_split_{split}') + '.csv'
    dataloader = Dataloader(
        raw_data_path='data/raw', 
        split_file=split_file,
        feature_data_path=f'data/features/CATCH22_TSB_{window_size}.csv',
        window_data_path=f'data/TSB_{window_size}',
    )
    if model_class == 'raw':
        data = dataloader.load_all_window_timeseries_parallel() if datasets is None else dataloader.load_multiple_window_timeseries_parallel(datasets)
    elif model_class == 'feature':
        data = dataloader.load_feature_all() if datasets is None else dataloader.load_feature_datasets(datasets)
    else:
        raise ValueError(f"Didn't recognize model class: {model_class}")
    data[data.isnull()] = 0

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read('VUS_PR')
    labels_df = metrics_df[detector.upper()]

    df_train, df_test = my_train_test_split(data.copy(), stratify=False, split_file=split_file)
    for df in [df_train, df_test]:
        df['label'] = np.array([labels_df.loc[x] for x in (df.index if window_size == 'entire' else ['.'.join(id.rsplit('.', maxsplit=1)[:-1]) for id in df.index])])
    y_train, x_train = df_train['label'].values, df_train.drop('label', axis=1).values
    y_test, x_test = df_test['label'].values, df_test.drop('label', axis=1).values

    # Train and evaluate the regressor
    curr_time = log_time(f"Training", tic, curr_time)
    toc = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - toc

    toc = time.time()
    y_pred = model.predict(x_test)
    inference_time = time.time() - toc
    y_pred = np.clip(y_pred, 0, 1)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    curr_time = log_time(f"Evaluation Metrics: MAE = {mae:.3f}, MSE: {mse:.3f}, R^2: {r2:.3f}", tic, curr_time)
    
    # Save the model and the results and training info (e.g. convnet_aeon_128_NORMA_supervised)
    with open(model_saving_path, 'wb') as f:
        pickle.dump(model, f)
    
    results_df = pd.DataFrame(
        data={'label': y_test, 'y_pred': y_pred},
        index=df_test.index
    )
    results_df.to_csv(results_path, index=True)

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
    metrics_df.to_csv(training_info_path, index=False)

    # How to load model
    # with open(model_saving_path, 'rb') as f:
    #     clf2 = pickle.load(f)
    # clf2.predict(x_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_regressors',
        description='Train any available regression model'
    )
    parser.add_argument('-m', '--model_index', type=str, help='index of the model architecture to train (from 0 to 28)', required=True)
    parser.add_argument('-d', '--detector', type=str, help='The detector\'s values you want to predict', required=True)
    parser.add_argument('-w', '--window_size', type=str, help='The size of the input subsequences', required=True)
    parser.add_argument('-e', '--experiment', type=str, help='Type of experiment (supervised or unsupervised)', required=True)
    parser.add_argument('-s', '--split', type=str, help='Which split to use, only for the unsupervised experiments', required=False, default=None)
    parser.add_argument('-p', '--saving_path', type=str, help='where to save the results of the experiment', required=True)
    parser.add_argument('-t', '--testing', type=bool, help='Testing or normal mode', required=False, default=False)
    
    args = parser.parse_args()

    if args.detector == 'all':
        args.detector = Scoreloader('data/scores').get_detector_names()
    else:
        args.detector = [args.detector]

    for detector in args.detector: 
        train_regressors(
            model_index=args.model_index,
            detector=detector,
            window_size=args.window_size,
            experiment=args.experiment,
            split=args.split,
            saving_path=args.saving_path,
            testing=args.testing,
        )