import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader


def metric_correlation():
    metric = 'AUC-PR'
    
    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric.replace('-', '_'))
    detectors_df = metrics_df[detectors]
    
    corr_matrix = detectors_df.corr(method='spearman')
    mask = (corr_matrix < 0.2) & (corr_matrix > -0.2)
    mask = False

    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, mask=mask, cmap="coolwarm")
    plt.show()

    # sns.pairplot(detectors_df, kind="kde")
    # plt.show()

        

if __name__ == "__main__":
    metric_correlation()