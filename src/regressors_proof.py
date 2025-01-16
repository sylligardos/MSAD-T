import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from combine_detectors import row_combine_detectors


def regressors_proof(dataset, saving_path):
    metric = 'AUC-PR'
    tqdm.pandas()
    
    dataloader = Dataloader('data/raw')
    if dataset == 'all':
        dataset = [x for x in dataloader.get_dataset_names() if 'nasa' not in x.lower()]
    x, y, fnames = dataloader.load_raw_datasets(dataset)
    
    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()
    scores, idx_failed = scoreloader.load_parallel(fnames)
    for idx in sorted(idx_failed, reverse=True):
        del fnames[idx]
        del y[idx]
        del x[idx]
    if (len(x) != len(y)) or (len(y) != len(fnames)):
        print(len(x), len(y), len(fnames))
        raise ValueError('Error when removing problematic timeseries')
    
    scores = dict(zip(fnames, scores))
    labels = dict(zip(fnames, y))

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric.replace('-', '_')).loc[fnames]
    detectors_df = metrics_df[detectors]
    oracle_df = metrics_df["TRUE_ORACLE-100"]
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Time series', 'TRUE_ORACLE-100': 'Oracle'})
    
    for k in range(2, 13):
        metrics_df[f"k={k}"] = metrics_df.progress_apply(lambda row: row_combine_detectors(row, scores, labels, detectors, k), axis=1)

    columns_to_keep = [x for x in metrics_df.columns if (x in detectors) or (x == "Oracle") or ('k=' in x) or (x == 'Time series')]
    curr_df = metrics_df[columns_to_keep]
    
    melted_curr_df = curr_df.melt(id_vars='Time series', var_name='Model', value_name=metric)
    
    order = melted_curr_df.groupby('Model')[metric].median().sort_values().index
    melted_curr_df['Model'] = pd.Categorical(melted_curr_df['Model'], categories=order, ordered=True)

    plt.figure(figsize=(16, 10))
    sns.boxplot(data=melted_curr_df, x='Model', y=metric, showmeans=True, whis=0.241289844, showfliers=False)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(saving_path, f'{dataset if not isinstance(dataset, list) else 'global'}.png'))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='regressors_proof',
        description='Prove whether combining detectors with regressors could work'
    )
    parser.add_argument('-d', '--dataset', type=str, help='index of the dataset to process (from 0 to 16)', required=True)
    parser.add_argument('-p', '--saving_path', type=str, help='where to save the results of the experiment', required=True)

    args = parser.parse_args()
    regressors_proof(dataset=args.dataset, saving_path=args.saving_path)