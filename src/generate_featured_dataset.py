"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD
"""

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader

import numpy as np
import pandas as pd
import argparse
import re
import os

from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.catch22 import Catch22


def generate_features(path, save_dir, feature_extractor, mode='reg'):
    """TODO

    :param path: path to the dataset to be converted
    """

    window_size = int(re.search(r'\d+', path).group())
    testing_flag = False

    # Create name of new dataset
    dataset_name = [x for x in path.split('/') if str(window_size) in x][0]
    new_name = f"{feature_extractor.upper()}_{dataset_name}.csv"

    # Load datasets
    dataloader = Dataloader("data/raw", window_data_path=f"data/TSB_{window_size}")
    datasets = dataloader.get_dataset_names() if not testing_flag else ['YAHOO']
    df_list = [dataloader.load_window_timeseries_parallel(dataset) for dataset in datasets]
    df = pd.concat(df_list)

    # Divide df
    labels = df.pop("label")
    x = df.to_numpy()[:, np.newaxis]
    index = df.index

    # Setup the TSFresh feature extractor (too costly to use any other parameter)
    if feature_extractor.lower() == 'tsfresh':
        fe = TSFreshFeatureExtractor(
            default_fc_parameters="minimal", 
            show_warnings=False, 
            n_jobs=-1
        )
    elif feature_extractor.lower() == 'catch22':
        fe = Catch22()
    else:
        raise ValueError(f"Not applicable feature extractor {feature_extractor}")
    
    # Compute features
    X_transformed = fe.fit_transform(x)

    # Create new dataframe
    X_transformed.index = index
    X_transformed = pd.merge(labels, X_transformed, left_index=True, right_index=True)
    
    # Save new features
    X_transformed.to_csv(os.path.join(save_dir, new_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_featured_dataset',
        description='Transform a dataset of raw time series to tabular data'
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
    parser.add_argument('-s', '--save_dir', type=str, help='where to save the generated dataset', required=True)
    parser.add_argument('-f', '--feature_extractor', type=str, help='which feature extractor to use', required=True)
    
    args = parser.parse_args()

    generate_features(
        path=args.path,
        save_dir=args.save_dir,
        feature_extractor=args.feature_extractor,
    )