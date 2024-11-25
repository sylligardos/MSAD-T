########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Spring Semester 2023
# @where : LIPADE / FORTH
# @title : ADecimo
# @component: models/utils
# @file : split_ts
#
########################################################################


import math
import numpy as np
import pandas as pd


def split_ts(data, window_size, step=None):
	'''Split a time series into windows according to window_size.
	If the time series can not be divided exactly by the window_size
	then the first window will overlap the second.

	:param data: the time series to be segmented
	:param window_size: the size of the windows
	:return data_split: a 2D array of the segmented time series
			indices: the list of starting indices of each window in the original time series
	'''
	if step is None or step == window_size:
		return split_ts_no_overlap(data, window_size)
	else:
		return split_ts_overlap(data, window_size, step)
	

def split_ts_overlap(data, window_size, step):
	'''Split a time series into windows according to window_size and step.
	If the time series cannot be divided exactly by the window_size
	and step, the first window may have a bigger overlap.

	How: Start from the end of the time series and 
	segment windows backwards.

	:param data: the time series to be segmented
	:param window_size: the size of the windows
	:param step: the step size (stride) between windows
	:return data_split: a 2D array of the segmented time series
	        indices: the list of starting indices of each window in the original time series
	'''
	if (step < 0) | (step > window_size):
		raise ValueError("Step should be between 0 and window_size")

	# Split the time series
	data_split = []
	indices = []
	i = len(data)
	while i >= window_size:
		data_split.append(data[i - window_size: i])
		indices.append(i - window_size)
		i -= step

	# Handle the first overlapping window
	if indices[-1]:
		data_split.append(data[:window_size])
		indices.append(0)

	data_split = np.asarray(data_split[::-1])
	indices = np.asarray(indices[::-1])

	return data_split, indices


def split_ts_no_overlap(data, window_size):
	'''Split a time series into windows according to window_size.
	If the time series can not be divided exactly by the window_size
	then the first window will overlap the second.

	:param data: the time series to be segmented
	:param window_size: the size of the windows
	:return data_split: a 2D array of the segmented time series
			indices: the list of starting indices of each window in the original time series
	'''
	if (window_size < 0) | window_size > data.shape[0]:
		raise ValueError(f'Window size {window_size} is not valid')

	# Compute the modulo
	modulo = data.shape[0] % window_size

	# Compute the number of windows
	k = data[modulo:].shape[0] / window_size

	# Split the time series
	data_split = np.split(data[modulo:], k)
	indices = np.arange(modulo, data.shape[0], window_size)

	# Handle first overlapping window
	if modulo != 0:
		data_split.insert(0, list(data[:window_size]))
		indices = np.insert(indices, 0, 0)

	return np.asarray(data_split), indices



