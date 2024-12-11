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



def log_time(message, start_time, curr_time):
    elapsed_time = time.time() - start_time
    current_elapsed_time = time.time() - curr_time
    print(f">> {message}, et: {elapsed_time:.2f} secs, cet: {current_elapsed_time:.2f} secs")
    return time.time()


def main():
    tic = time.time()
    curr_time = tic
    window_size = 128
    # datasets = ['YAHOO']
    datasets = None
    metric = 'VUS_PR'
    
    # Read the raw time series, anomalies
    curr_time = log_time(f"Loading data", tic, curr_time)
    dataloader = Dataloader(
        raw_data_path='data/raw', 
        split_file=f'data/splits/supervised/split_TSB.csv', 
        feature_data_path='data/features/CATCH22_TSB_128.csv'
    )
    data = dataloader.load_feature_all() if datasets is None else dataloader.load_feature_datasets(datasets)

    # Load the scores of all detectors
    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    # Load metrics
    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric)
    detectors_df = metrics_df[detectors]

    # Setup current labels
    # labels_df = detectors_df[curr_detector]
    # index = list(data.index)
    # index_without_number = ['.'.join(id.rsplit('.', maxsplit=1)[:-1]) for id in index]
    # labels = np.array([labels_df.loc[x] for x in index_without_number])
    # data['label'] = labels

    # Prepare the data for training
    curr_time = log_time("Splitting the data into subsets", tic, curr_time)
    df_train, df_test = my_train_test_split(data, stratify=False, split_file="data/splits/supervised/split_TSB.csv")
    y_train, x_train = df_train['label'], df_train.drop('label', axis=1)
    y_test, x_test = df_test['label'], df_test.drop('label', axis=1)

    # Loading all models
    curr_time = log_time("Loading all the models", tic, curr_time)
    models = {}
    for detector in detectors:
        models[detector] = xgboost.XGBRegressor()
        models[detector].load_model(os.path.join("experiments", "regression_def_11_12_2024", "models", f"{detector}_{window_size}_reg_xgboost.json"))

    # Predict on the test set
    curr_time = log_time("Predicting the test set", tic, curr_time)
    all_predictions = []
    for detector in detectors:
        y_pred = models[detector].predict(x_test)
        y_pred_df = pd.DataFrame(y_pred, index=x_test.index, columns=[f'{detector}'])
        y_pred_df[y_pred_df < 0] = 0
        y_pred_df[y_pred_df > 1] = 1
        y_pred_df['Time series'] = y_pred_df.index.to_series().apply(lambda idx: '.'.join(idx.rsplit('.', maxsplit=1)[:-1]))
    
        # Combine results of all windows for every time series
        agg_method = 'mean' #['mean', 'median', 'gmean', 'hmean', 'mode', 'percentile_25', 'percentile_75']
        grouped_predictions = y_pred_df.groupby('Time series')[f'{detector}'].agg(agg_method)
        all_predictions.append(grouped_predictions)
    results = pd.concat(all_predictions, axis=1)
    
    results['Selected detector'] = results.idxmax(axis=1)
    results['Selected VUS-PR'] = results.apply(lambda row: detectors_df.loc[row.name][row['Selected detector']], axis=1)
    results['Oracle'] = results.apply(lambda row: detectors_df.loc[row.name].max(), axis=1)
    results['Dataset'] = results.index.to_series().apply(lambda idx: idx.split('/', maxsplit=1)[0])
    results_melted = results[['Selected VUS-PR', 'Oracle']].melt(var_name='Metric', value_name='Value')
    
    print(Counter(results['Dataset']))
    print(results)

    # Create the figure with a larger height
    plt.figure(figsize=(6, 10))
    sns.boxplot(data=results_melted, y='Value', x='Metric', showmeans=True, whis=0.241289844)
    plt.yticks(np.linspace(0, 1, num=11))  # Increase the number of y-ticks
    plt.grid(axis='y')
    plt.show()
    

if __name__ == "__main__":
    main()