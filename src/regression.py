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



def log_time(message, start_time, curr_time):
    elapsed_time = time.time() - start_time
    current_elapsed_time = time.time() - curr_time
    print(f">> {message}, et: {elapsed_time:.2f} secs, cet: {current_elapsed_time:.2f} secs")
    return time.time()


def main(curr_detector):
    tic = time.time()
    curr_time = tic
    window_size = 128
    # datasets = ['YAHOO']
    datasets = None
    metric = 'VUS_PR'
    
    # Read the raw time series, anomalies
    curr_time = log_time(f"-------- Detector: {curr_detector} window size: {window_size}. Loading data", tic, curr_time)
    dataloader = Dataloader(
        raw_data_path='data/raw', 
        split_file=f'data/splits/supervised/split_TSB_{window_size}.csv', 
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
    labels_df = detectors_df[curr_detector]
    index = list(data.index)
    index_without_number = ['.'.join(id.rsplit('.', maxsplit=1)[:-1]) for id in index]
    labels = np.array([labels_df.loc[x] for x in index_without_number])
    data['label'] = labels

    # Prepare the data for training
    curr_time = log_time("Splitting the data into subsets", tic, curr_time)
    df_train, df_test = my_train_test_split(data, stratify=False, split_file="data/splits/supervised/split_TSB.csv")
    y_train, x_train = df_train['label'], df_train.drop('label', axis=1)
    y_test, x_test = df_test['label'], df_test.drop('label', axis=1)

    # Train Regressor
    curr_time = log_time("Training the classifier", tic, curr_time)
    # model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, verbosity=1)
    model = xgboost.XGBRegressor()
    model.fit(x_train, y_train)

    # Predict on the test set
    curr_time = log_time("Predicting the test set", tic, curr_time)
    y_pred = model.predict(x_test)
    
    # Combine windows' results to get results per timeseries
    curr_time = log_time("Combining the predictions of the test set", tic, curr_time)
    y_pred_df = pd.DataFrame(y_pred, index=x_test.index, columns=['y_pred'])
    y_pred_df[y_pred_df < 0] = 0
    y_pred_df[y_pred_df > 1] = 1
    y_pred_df['Time series'] = y_pred_df.index.to_series().apply(lambda idx: '.'.join(idx.rsplit('.', maxsplit=1)[:-1]))
    y_pred_df

    # Aggregation methods
    aggregation_methods = {
        'mean': np.mean,
        'median': np.median,
        'gmean': lambda x: stats.mstats.gmean(x),
        'hmean': lambda x: stats.hmean(x),
        'mode': lambda x: stats.mode(x.round(1))[0],
        'percentile_25': lambda x: np.percentile(x, 25),
        'percentile_75': lambda x: np.percentile(x, 75),
        'range': np.ptp,
    }
    grouped_predictions = y_pred_df.groupby('Time series')['y_pred'].apply(aggregation_methods['mean'])

    # Store the results
    test_results = pd.DataFrame({
        'Time series': grouped_predictions.index,
        metric: grouped_predictions.index.map(labels_df),
        'y_pred': grouped_predictions.values
    }).reset_index(drop=True).sort_values(by='Time series')

    # Evaluate the model for each aggregation method
    mae = mean_absolute_error(test_results[metric], test_results['y_pred'])
    mse = mean_squared_error(test_results[metric], test_results['y_pred'])
    r2 = r2_score(test_results[metric], test_results['y_pred'])

    # Print the metrics for each method
    print(f"Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Saving the model
    curr_time = log_time("Saving the model", tic, curr_time)
    model.save_model(os.path.join("experiments", "regression_def_11_12_2024", "models", f"{curr_detector}_{window_size}_reg_xgboost.json"))


if __name__ == "__main__":
    detectors = Scoreloader(scores_path='data/scores').get_detector_names()
    for detector in detectors:
        main(curr_detector=detector)