"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

import argparse


def train_regressors(
        model,
        detector,
        window_size,
        experiment
    ):
    print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_regressors',
        description='Train any available regression model'
    )
    parser.add_argument('-m', '--model', type=str, help='model architecture to train', required=True)
    parser.add_argument('-d', '--detector', type=str, help='The detector\'s values you want to predict', required=True)
    parser.add_argument('-w', '--window_size', type=str, help='The size of the input subsequences', required=True)
    parser.add_argument('-e', '--experiment', type=str, help='Type of experiment (supervised or unsupervised)', required=True)
    
    args = parser.parse_args()

    train_regressors(
        model=args.model,
        detector=args.detector,
        window_size=args.window_size,
        experiment=args.experiment,
    )