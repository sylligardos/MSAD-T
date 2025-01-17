from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from data.dataloader import Dataloader
from utils.norm import z_normalization
from utils.split_ts import split_ts, my_train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from collections import Counter
import pickle



def log_time(message, start_time, curr_time):
    elapsed_time = time.time() - start_time
    current_elapsed_time = time.time() - curr_time
    print(f">> {message}, et: {elapsed_time:.2f} secs, cet: {current_elapsed_time:.2f} secs")
    return time.time()


def main():
    tic = time.time()
    curr_time = tic
    metric = 'VUS_PR'
    window_size = 'entire'

    # datasets = ['YAHOO']
    datasets = None
    dataloader = Dataloader(
        raw_data_path='data/raw', 
        split_file=f'data/splits/supervised/split_TSB.csv', 
        feature_data_path=f'data/features/CATCH22_TSB_{window_size}.csv'
    )
    data = dataloader.load_feature_all() if datasets is None else dataloader.load_feature_datasets(datasets)

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric)
    detectors_df = metrics_df[detectors]

    models = {}
    for detector in detectors:
        model_path = os.path.join("experiments", "regression_13_01_2025", "models", f"XGBRegressor_feature_{window_size}_{detector}_supervised.json")
        with open(model_path, 'rb') as f:
            models[detector] = pickle.load(f)

    all_predictions = []
    for detector in detectors:
        y_pred = models[detector].predict(data)
        y_pred = np.clip(y_pred, 0, 1)
        y_pred_df = pd.DataFrame(y_pred, index=data.index, columns=[f'{detector}'])
        if window_size != "entire":
            y_pred_df['Time series'] = y_pred_df.index.to_series().apply(lambda idx: '.'.join(idx.rsplit('.', maxsplit=1)[:-1]))
            grouped_predictions = y_pred_df.groupby('Time series')[f'{detector}'].agg('mean')
            all_predictions.append(grouped_predictions)
        else:
            all_predictions.append(y_pred_df)
    results = pd.concat(all_predictions, axis=1)

    save_path = os.path.join("experiments", "regression_13_01_2025", "inference", f"XGBRegressor_feature_{window_size}_supervised.csv")
    results.to_csv(save_path)


if __name__ == "__main__":
    main()