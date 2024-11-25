"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MySCAM
"""


from utils.utils import load_timeseries, load_anomaly_scores, load_model, plot_signal_with_anomalies
from data.scoreloader import Scoreloader
from data.metricloader import Metricloader
from utils.norm import z_normalization

import torch
import torch.nn as nn
from torchinfo import summary
from utils.CAM import CAM
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import numpy as np


def main():
	# Setup
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	tensor_softmax = nn.Softmax(dim=1)
	scoreloader = Scoreloader('data/scores')
	detectors = scoreloader.get_detector_names()

	# Read the raw time series, anomalies
	timeseries, anomalies, fnames, df = load_timeseries(raw_data_path='data/raw', window_data_path='data/TSB_128')
	timeseries = timeseries[:10]
	anomalies = anomalies[:10]
	fnames = fnames[:10]

	# Load the scores of all detectors
	scores, idx_failed = load_anomaly_scores(path="data/scores", fnames=fnames)

	# Load the label of each time series
	df = df.reset_index(names='Time series')
	label_df = df[['Time series', 'label']].copy()
	label_df['Time series'] = label_df['Time series'].apply(lambda x: '.'.join(x.split('.')[:-1]))
	label_df = label_df.drop_duplicates(subset=['Time series']).set_index('Time series')
	
	# Load metrics
	metricloader = Metricloader('data/metrics')
	metrics_df = metricloader.read('AUC_PR')
	
	# Load convnet
	conv_model = load_model(
		model_name="convnet", 
		window_size=128, 
		weights_path="models/supervised", 
	)
	conv_model.eval()

	# Change layers of convnet to inference the whole timeseries
	layers = list(conv_model.children())
	model = nn.Sequential(layers[0], nn.AdaptiveAvgPool1d(1), nn.Flatten(), layers[-1])
	model.eval()
	# summary(model, input_size=(1, 1, 128), col_names=['input_size', 'output_size'])

	# Get CAM (Class Activation Map) object
	# [print(name, layer, '\n------') for name, layer in model.named_modules()]
	cam = CAM(model, device, last_conv_layer=model._modules['0']._modules['15'], fc_layer_name=model._modules['3']._modules['0'])

	# Inference raw time series and get CAM for every class
	preds = []
	for i, (x, y) in enumerate(zip(timeseries, anomalies)):
		curr_label = int(label_df.loc[fnames[i]].values[0])

		# Normalize time series
		x = z_normalization(x)

		# Get cam and predictions
		curr_cams, pred = cam.run(x.reshape(1, -1))
		preds.append(pred)
		curr_cams[curr_cams < 0] = 0

		# Plot results
		fig, ax = plt.subplots(13, 1, sharex=True, figsize=(10, 10))
		plot_signal_with_anomalies(x, y, title=None, ax=ax[0])
		
		for (j, detector), prob in zip(enumerate(detectors), pred):
			sns.lineplot(ax=ax[1 + j], data=curr_cams[j])
			ax[1 + j].set_ylabel(f"{detector}\n{prob:.2f}\n{np.mean(curr_cams[j]):.2f}\n{metrics_df.loc[fnames[i], detector]:.2f}")
			ax[1 + j].sharey(ax[1 + pred.argmax()])

		plt.suptitle(f"Best {detectors[curr_label]} Pred {detectors[pred.argmax()]} for {fnames[i]}")
		plt.show()
		

if __name__ == "__main__":
	main()
