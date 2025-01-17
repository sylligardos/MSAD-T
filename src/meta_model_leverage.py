from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from data.dataloader import Dataloader

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class AdjusterModel(nn.Module):
    def __init__(self):
        super(AdjusterModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(12, 64),  # Input: 12 scores
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12)   # Output: 12 adjusted scores
        )

    def forward(self, x):
        return self.fc(x)
    
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        for preds, targets in dataloader:
            adjusted_scores = model(preds)  # Forward pass
            loss = criterion(adjusted_scores, targets)  # Compute loss
            total_loss += loss.item() * preds.size(0)  # Accumulate loss
            all_preds.append(adjusted_scores.cpu().numpy())  # Store predictions
            all_targets.append(targets.cpu().numpy())  # Store targets

    # Combine all predictions and targets for final evaluation
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mean_loss = total_loss / len(dataloader.dataset)  # Average loss
    return mean_loss, all_preds, all_targets
    

class ScoresDataset(Dataset):
    def __init__(self, predicted_scores, true_scores):
        """
        Args:
            predicted_scores (numpy.ndarray or torch.Tensor): Array of shape (num_samples, 12).
            true_scores (numpy.ndarray or torch.Tensor): Array of shape (num_samples, 12).
        """
        self.predicted_scores = torch.tensor(predicted_scores, dtype=torch.float32)
        self.true_scores = torch.tensor(true_scores, dtype=torch.float32)

    def __len__(self):
        return len(self.predicted_scores)

    def __getitem__(self, idx):
        # Returns a tuple (predicted scores, true scores) for a given index
        return self.predicted_scores[idx], self.true_scores[idx]


def main():
    metric = 'VUS-PR'
    window_size = 'entire'

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()

    metricloader = Metricloader('data/metrics')
    metrics_df = metricloader.read(metric.replace('-', '_'))
    detectors_df = metrics_df[detectors]

    split_path = os.path.join('data', 'splits', 'supervised', 'split_TSB.csv')
    splits = pd.read_csv(split_path, index_col=0)
    train_set = [x[:-len('.csv')] for x in splits.loc['train_set']]
    test_set = [x[:-len('.csv')] for x in splits.loc['val_set'].dropna()]
    
    results_path = os.path.join("experiments", "regression_13_01_2025", "inference", f"XGBRegressor_feature_{window_size}_supervised.csv")
    df = pd.read_csv(results_path, index_col=0)
    x_train = df.loc[train_set]
    y_train = detectors_df.loc[train_set]
    
    x_test = df.loc[test_set]
    y_test = detectors_df.loc[test_set]

    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.33)

    batch_size = 32
    train_dataset = ScoresDataset(x_train.values, y_train.values)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ScoresDataset(x_test.values, y_test.values)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = AdjusterModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            preds, targets = batch
            preds, targets = preds.float(), targets.float()
            
            optimizer.zero_grad()
            adjusted_scores = model(preds)
            loss = criterion(adjusted_scores, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    test_loss, y_pred, _ = evaluate_model(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    y_pred = pd.DataFrame(data=np.clip(y_pred, 0, 1), columns=detectors, index=y_test.index)

    mae = mean_absolute_error(y_test, x_test)
    mse = mean_squared_error(y_test, x_test)
    r2 = r2_score(y_test, x_test)

    print(mae)
    print(mse)
    print(r2)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(mae)
    print(mse)
    print(r2)

    x_test['Selected detector'] = x_test[detectors].idxmax(axis=1)
    x_test[f'Selected {metric} before'] = x_test.apply(lambda row: detectors_df.loc[row.name][row['Selected detector']], axis=1)
    x_test['Oracle'] = x_test.apply(lambda row: detectors_df.loc[row.name].max(), axis=1)
    
    y_pred['Selected detector'] = y_pred[detectors].idxmax(axis=1)
    y_pred[f'Selected {metric} after'] = y_pred.apply(lambda row: detectors_df.loc[row.name][row['Selected detector']], axis=1)

    comparison_df = pd.concat([x_test[f'Selected {metric} before'], y_pred[f'Selected {metric} after'], x_test['Oracle']], axis=1)
    sns.boxplot(comparison_df, showmeans=True, whis=0.241289844)
    plt.grid(axis='y')
    plt.show()

    # Plot predictions vs ground truth for a specific detector
    # detector_idx = 0
    # plt.scatter(y_test.iloc[:, detector_idx], x_test.iloc[:, detector_idx], alpha=0.5, label='before')
    # plt.scatter(y_test.iloc[:, detector_idx], y_pred[:, detector_idx], alpha=0.5, label='before')
    # plt.xlabel('Ground Truth Scores')
    # plt.ylabel('Predicted Scores')
    # plt.title(f'Detector {detector_idx+1}: Predictions vs Ground Truth')
    # plt.show()




if __name__ == "__main__":
    main()