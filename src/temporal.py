"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MySCAM
"""


from utils.utils import load_anomaly_scores, load_model, plot_signal_with_anomalies, detect_boundaries, compute_temporal_score, compute_metrics
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from data.dataloader import Dataloader
from utils.norm import z_normalization
from utils.split_ts import split_ts

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
	timeseries, anomalies, fnames = dataloader.load_raw_datasets(dataloader.get_dataset_names() if datasets_to_process is None else datasets_to_process, split='val')
	
	# Uncomment for testing
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
	
	# Inference the dataset
	results = []
	for i, (x, y, fname, curr_scores) in tqdm(enumerate(zip(timeseries, anomalies, fnames, scores)), desc='Processing ts', total=len(fnames)):
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
		# probs = tensor_softmax(pred).detach().numpy()
		probs = tensor_softmax(pred).detach().cpu().numpy()

		# Compute temporal score
		combined_score = compute_temporal_score(curr_scores, probs, indices, window_size)
		if not os.path.exists(os.path.join(scores_saving_dir, curr_dataset)):
			os.makedirs(os.path.join(scores_saving_dir, curr_dataset))
		np.savetxt(os.path.join(results_dir, 'scores', f"{model_name}_{window_size}_{step}", f"{fname}.csv"), combined_score, delimiter=",")

		# Compute AUC-PR of combined score
		curr_metrics = compute_metrics([y], combined_score[np.newaxis, :], k=None)[0]
		
		# Save results
		results.append({
			"Time series": fname,
			"Dataset": curr_dataset,
		} | curr_metrics)
		
		if flag_visualize_results:
			# Find out significant probs regions :D (Not used currently)
			boundaries, bad_regions = detect_boundaries(probs, threshold=0.5)

			# Find most predicted detector (MVP)
			mean_probs = probs.mean(axis=0)
			top_k_detectors = np.argsort(mean_probs)[-3:]

			# Compute entropy of probabilities per window
			ent = entropy(probs, axis=1)

			# Plot results
			fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
			
			plot_signal_with_anomalies(x, y, x_axis=new_x_axis, title=None, ax=ax[0])
			# ax[0].set_xlim(0, len(x)-1)
			ax[0].set_title(f'{fname}')

			sns.heatmap(
				probs.T, 
				cbar=False, 
				yticklabels=detectors, 
				ax=ax[1]
			)
			ax[1].set_title(f'Probabilities')
			
			# bad_regions_y = [boundaries[x] for x in bad_regions]
			sns.scatterplot(boundaries, ax=ax[2])
			sns.scatterplot(x=bad_regions, y=[boundaries[x] for x in bad_regions], ax=ax[2], color='red')
			ax[2].set_xlim(0, len(probs) - 1)
			ax[2].set_yticks(ticks=np.arange(0, len(detectors)), labels=detectors)
			ax[2].set_title('Decision boundaries')
			ax[2].invert_yaxis()
			ax[2].grid(axis='y')

			for j in top_k_detectors:
				ax[3].plot(new_x_axis, curr_scores[:, j], label=f"{detectors[j]}", alpha=0.5)
			ax[3].legend()
			# ax[3].set_xlim(0, len(x)-1)
			ax[3].set_ylim(0, 1)
			ax[3].set_title("MVP's scores")
			# ax[3].set_xticks(ticks=[])
			
			# To print
			pd.set_option('display.float_format', '{:.3f}'.format)
			curr_df.columns = ['Score']
			curr_df['Probability'] = mean_probs

			plt.suptitle(f'Best detector {curr_label}; Predicted {detectors[mean_probs.argmax()]}')
			plt.tight_layout()
			plt.show()

	final_df = pd.DataFrame(results).set_index('Time series')
	saving_path = os.path.join(results_dir, "results", f"{model_name}_{window_size}_{step}.csv")
	final_df.to_csv(saving_path)

if __name__ == "__main__":
	# Experiment's parameters
	window_size = 128
	step = window_size//2
	model_name = 'convnet'
	datasets_to_process = None
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
