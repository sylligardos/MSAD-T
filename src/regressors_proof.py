import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from collections import Counter
from pandarallel import pandarallel

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from combine_detectors import combine_detectors, row_combine_detectors


def regressors_proof():
    save_path = os.path.join('experiments', 'regressors_proof')
    metric = 'AUC-PR'
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    
    dataloader = Dataloader('data/raw')
    datasets = dataloader.get_dataset_names()
    for curr_dataset in datasets[0:2]:
        x, y, fnames = dataloader.load_raw_datasets(curr_dataset)
        
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
            metrics_df[f"k={k}"] = metrics_df.parallel_apply(lambda row: row_combine_detectors(row, scores, labels, detectors, k), axis=1)

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
        plt.savefig(os.path.join(save_path, f'{curr_dataset}.png'))
    
    
    
    
            

if __name__ == "__main__":
    regressors_proof()
