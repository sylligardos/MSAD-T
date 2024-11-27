"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MySCAM
"""


from utils.utils import load_anomaly_scores, load_model, plot_signal_with_anomalies, wrapper_get_metrics, compute_temporal_score, compute_metrics
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from data.dataloader import Dataloader
from utils.norm import z_normalization
from utils.split_ts import split_ts

import argparse
from tqdm import tqdm
import os
from scipy.stats import entropy
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def main(
	window_size,
	step,
	model_name,
	datasets_to_process,
	results_dir,
	flag_visualize_results,	
	testing,
):
	# Setup
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	tensor_softmax = nn.Softmax(dim=1)
	scores_saving_dir = os.path.join(results_dir, 'scores', f"{model_name}_{window_size}_{step}")
	
	# Read the raw time series, anomalies
	dataloader = Dataloader(raw_data_path='data/raw', split_file=f'data/splits/supervised/split_TSB_{window_size}.csv')
	timeseries, anomalies, fnames = dataloader.load_raw_datasets(
		dataloader.get_dataset_names() if datasets_to_process is None else datasets_to_process, split='val'
	)
	
	# For testing
	if testing:
		random_choices = np.random.choice(np.arange(0, len(timeseries)), size=10, replace=False)
		timeseries = [timeseries[x] for x in random_choices]
		anomalies = [anomalies[x] for x in random_choices]
		fnames = [fnames[x] for x in random_choices]

	# Load the scores of all detectors
	scoreloader = Scoreloader('data/scores')
	detectors = scoreloader.get_detector_names()
	scores, idx_failed = load_anomaly_scores(path="data/scores", fnames=fnames)
	for idx in sorted(idx_failed, reverse=True):
		del fnames[idx]
		del anomalies[idx]
		del timeseries[idx]

	# Load metrics
	metricloader = Metricloader('data/metrics')
	metrics_df = metricloader.read('AUC_PR')
	detectors_df = metrics_df[detectors]
	
	# Load convnet
	model = load_model(
		model_name=model_name, 
		window_size=window_size, 
		weights_path="models/supervised", 
	)
	model.eval()
	model.to(device)

	# Prepare args
	args = [
		(x, y, detectors_df, fname, model, tensor_softmax, device, curr_scores, scores_saving_dir, detectors, flag_visualize_results)
		for (x, y, fname, curr_scores) in zip(timeseries, anomalies, fnames, scores)
	]
	
	# Inference the dataset
	results = []
	for arg in tqdm(args, desc="Inference"):
		result = process_timeseries(arg)
		results.append(result)

	final_df = pd.DataFrame(results).set_index('Time series')
	saving_path = os.path.join(results_dir, "results", f"{model_name}_{window_size}_{step}.csv")
	final_df.to_csv(saving_path)


def process_timeseries(args):
	# Unbox args
	x, y, detectors_df, fname, model, tensor_softmax, device, curr_scores, scores_saving_dir, detectors, flag_visualize_results = args
	
	curr_label = detectors_df.loc[fname].idxmax()
	curr_df = pd.DataFrame(detectors_df.loc[fname])
	curr_dataset = fname.split('/')[0]

	# Normalize time series
	x = z_normalization(x)

	# Split time series
	x_seg, indices = split_ts(x, window_size, step=step)
	new_x_axis = np.linspace(0, len(x)/(len(x)/len(x_seg)), len(x)) # used for plotting

	# Get predictions
	data = torch.from_numpy(x_seg[:, np.newaxis]).to(device)
	pred = model(data.float())
	probs = tensor_softmax(pred).detach().numpy()

	# Compute temporal score
	combined_score, prob_mask = compute_temporal_score(curr_scores, probs, indices, window_size)
	if not os.path.exists(os.path.join(scores_saving_dir, curr_dataset)):
		os.makedirs(os.path.join(scores_saving_dir, curr_dataset))
	np.savetxt(os.path.join(results_dir, 'scores', f"{model_name}_{window_size}_{step}", f"{fname}.csv"), combined_score, delimiter=",")

	# Compute AUC-PR of combined score
	curr_metrics = wrapper_get_metrics((combined_score, y))
	
	# Save results
	result = {
		"Time series": fname,
		"Dataset": curr_dataset,
	} | curr_metrics
	
	# Plot results
	if flag_visualize_results:
		fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
		
		plot_signal_with_anomalies(x, y, x_axis=new_x_axis, title=None, ax=ax[0])
		ax[0].set_title(f'{fname}')

		sns.heatmap(
			probs.T, 
			cbar=False, 
			yticklabels=detectors, 
			ax=ax[1],
		)
		ax[1].set_title('Probabilities')
		
		for i, detector_name in enumerate(detectors):
			sns.lineplot(x=new_x_axis, y=prob_mask[:, i], ax=ax[2], label=f"{detector_name}: {curr_df.loc[detector_name].values[0]:.2f}")
		ax[2].set_title('Probabilities mask')
		ax[2].grid(axis='y')
		ax[2].legend(title="Detectors", ncol=2)

		sns.lineplot(x=new_x_axis, y=combined_score, ax=ax[3])
		ax[3].set_xticks([])
		ax[3].set_title(f'Combine score; AUC-PR: {curr_metrics['AUC-PR']:.2f}')

		plt.suptitle(f'Best detector {curr_label}')
		plt.tight_layout()
		plt.show()

	return result

if __name__ == "__main__":
	# CMD-line arguments
	parser = argparse.ArgumentParser(description="Run MSAD-T experiments")
	parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
	parser.add_argument('--window_size', type=int, required=True, help='Window size to segment time series')
	parser.add_argument('--step', type=int, required=True, help='Step of the window (can be overlapping or not)')

	args = parser.parse_args()

	# Experiment's parameters
	window_size = args.window_size
	step = args.step
	model_name = args.model_name
	datasets_to_process = "YAHOO"
	results_dir = "experiments/25_11_2024"
	flag_visualize_results = False
	testing = False
	
	main(
		window_size=window_size,
		step=step,
		model_name=model_name,
		datasets_to_process=datasets_to_process,
		results_dir=results_dir,
		flag_visualize_results=flag_visualize_results,
		testing=testing,
	)