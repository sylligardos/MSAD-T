import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from collections import Counter

from data.scoreloader import Scoreloader
from data.metricloader import Metricloader


def eval_regressors():
    results_path = os.path.join('experiments', 'regression_08_01_2025', 'results')

    result_files = [file for file in os.listdir(results_path) if file.endswith('.csv')]
    supervised_files = [file for file in result_files if "_supervised" in file]
    unsupervised_files = [file for file in result_files if "_unsupervised_" in file]

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read("VUS_PR")
    detectors_df = metrics_df[detectors]

    # Load supervised results
    supervised_results_list = []
    # supervised_files = [file for file in supervised_files if ('XGBRegressor_feature_128' in file) or ('XGBRegressor_feature_512' in file)]  # Test
    for file in tqdm(supervised_files):
        parts = file.split('_')
        
        if len(parts) != 5:
            print("STOP! THERE IS A MISTAKE WITH GROUPING THE FILES. FILE NAME HAS PROBABLY CHANGED. TAKE CARE")
        else:
            curr_df = pd.read_csv(os.path.join(results_path, file), index_col=0)
            curr_df['Time series'] = curr_df.index.to_series().apply(lambda idx: '.'.join(idx.rsplit('.', maxsplit=1)[:-1]))
            curr_mean_df = curr_df.groupby('Time series').agg('mean')
            curr_mean_df['Regressor name'] = parts[0]
            # curr_mean_df['Regressor type'] = parts[1]
            curr_mean_df['Window size'] = parts[2]
            curr_mean_df['Detector'] = parts[3]
            supervised_results_list.append(curr_mean_df)

    supervised_results_df = pd.concat(supervised_results_list)
    supervised_results_df.reset_index(inplace=True)
    regressor_df = supervised_results_df[['Regressor name', 'Window size']].drop_duplicates().reset_index(drop=True)

    # Analyse for every regressor
    # for _, row in regressor_df.iterrows():
    # curr_df = supervised_results_df[(supervised_results_df['Regressor name'] == row['Regressor name']) & (supervised_results_df['Window size'] == row['Window size'])].drop(columns=['Regressor name', 'Window size'])
    curr_df = supervised_results_df
    curr_df['Model name'] = curr_df['Regressor name'] + '_' + curr_df['Window size']
    curr_df = curr_df.drop(columns=['Regressor name', 'Window size'])
    curr_df['Error'] = np.abs(curr_df['label'] - curr_df['y_pred'])
    
    curr_piv_df = curr_df.pivot_table(
        index=['Time series', 'Model name'],
        columns='Detector',
        values='y_pred',
    ).reset_index()
    curr_piv_df['Selected detector'] = curr_piv_df[detectors].idxmax(axis=1)
    curr_piv_df['Selected VUS-PR'] = curr_piv_df.apply(lambda row: detectors_df.loc[row['Time series']][row['Selected detector']], axis=1)
    curr_piv_df['Oracle'] = curr_piv_df.apply(lambda row: detectors_df.loc[row['Time series']].max(), axis=1)
    curr_piv_df['Dataset'] = curr_piv_df['Time series'].apply(lambda idx: idx.split('/', maxsplit=1)[0])
    # curr_df_melted = curr_piv_df[['Selected VUS-PR', 'Oracle']].melt(var_name='Metric', value_name='Value')
    
    print(curr_piv_df)
    plt.figure(figsize=(6, 10))
    sns.boxplot(data=curr_piv_df, y='Selected VUS-PR', x='Model name', showmeans=True, whis=0.241289844)
    plt.yticks(np.linspace(0, 1, num=11))  # Increase the number of y-ticks
    plt.grid(axis='y')
    plt.show()
            

if __name__ == "__main__":
    eval_regressors()


def load_results(csv_file):
    return pd.read_csv(csv_file, index_col=0)