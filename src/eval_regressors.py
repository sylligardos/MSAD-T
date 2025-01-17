import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from collections import Counter

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from combine_detectors import combine_detectors


def eval_regressors():
    metric = 'VUS-PR'
    results_path = os.path.join('experiments', 'xgbregressors_17_01_2025', 'results')

    result_files = [file for file in os.listdir(results_path) if file.endswith('.csv')]
    supervised_files = [file for file in result_files if "_supervised" in file]
    unsupervised_files = [file for file in result_files if "_unsupervised_" in file]

    dataloader = Dataloader('data/raw')

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric.replace('-', '_'))
    detectors_df = metrics_df[detectors]

    # Load supervised results
    supervised_results_list = []
    supervised_files = [file for file in supervised_files]  # Test  if ('RandomIntervalRegressor' in file)
    for file in tqdm(supervised_files):
        parts = file.split('_')
        if len(parts) != 5:
            raise ValueError("STOP! THERE IS A MISTAKE WITH GROUPING THE FILES. FILE NAME HAS PROBABLY CHANGED.")

        curr_df = pd.read_csv(os.path.join(results_path, file), index_col=0)
        if parts[2] == 'entire':
            curr_df.index.names = ['Time series']
            curr_mean_df = curr_df
        else:
            curr_df['Time series'] = curr_df.index.to_series().apply(lambda idx: '.'.join(idx.rsplit('.', maxsplit=1)[:-1]))
            curr_mean_df = curr_df.groupby('Time series').agg('mean')
        curr_mean_df['Regressor name'] = parts[0]
        curr_mean_df['Regressor type'] = parts[1]
        curr_mean_df['Window size'] = parts[2]
        curr_mean_df['Detector'] = parts[3]
        supervised_results_list.append(curr_mean_df)

    supervised_results_df = pd.concat(supervised_results_list)
    supervised_results_df.reset_index(inplace=True)
    regressor_df = supervised_results_df[['Regressor name', 'Window size']].drop_duplicates().reset_index(drop=True)

    # Analyse for every regressor
    curr_df = supervised_results_df
    curr_df['Model name'] = curr_df['Regressor name'] + '-' + curr_df['Window size']
    curr_df = curr_df.drop(columns=['Regressor name', 'Window size'])
    curr_df['Error'] = np.abs(curr_df['label'] - curr_df['y_pred'])
    
    curr_piv_df = curr_df.pivot_table(
        index=['Time series', 'Model name'],
        columns='Detector',
        values='y_pred',
    ).reset_index()
    curr_piv_df['Selected detector'] = curr_piv_df[detectors].idxmax(axis=1)
    curr_piv_df[f'Selected {metric}'] = curr_piv_df.apply(lambda row: detectors_df.loc[row['Time series']][row['Selected detector']], axis=1)
    curr_piv_df['Oracle'] = curr_piv_df.apply(lambda row: detectors_df.loc[row['Time series']].max(), axis=1)
    curr_piv_df['Dataset'] = curr_piv_df['Time series'].apply(lambda idx: idx.split('/', maxsplit=1)[0])
        
    # scores, idx_failed = scoreloader.load_parallel(curr_piv_df['Time series'].unique())
    # scores = dict(zip(curr_piv_df['Time series'].unique(), scores))
    
    # _, labels, fnames = dataloader.load_timeseries_parallel(curr_piv_df['Time series'].unique())
    # labels = dict(zip(fnames, labels))

    # new_df = combine_detectors(curr_piv_df[curr_piv_df['Time series'].isin(fnames)], scores, labels, detectors)
    # print(new_df)
    # exit()
    # curr_piv_df = new_df

    print(curr_piv_df)
    order = curr_piv_df.groupby('Model name')[f'Selected {metric}'].median().sort_values().index
    curr_piv_df['Model name'] = pd.Categorical(curr_piv_df['Model name'], categories=order, ordered=True)
    plt.figure(figsize=(6, 10))
    sns.boxplot(data=curr_piv_df, y=f'Selected {metric}', showmeans=True, x='Model name', whis=0.241289844, showfliers=False)
    # sns.boxplot(data=curr_piv_df, y=f'k=4', showmeans=True, whis=0.241289844, showfliers=False)
    # sns.boxplot(data=curr_piv_df, y='Oracle', showmeans=True, whis=0.241289844, showfliers=False)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
            

if __name__ == "__main__":
    eval_regressors()


def load_results(csv_file):
    return pd.read_csv(csv_file, index_col=0)