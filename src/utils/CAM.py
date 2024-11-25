import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk


class SaveFeatures():
	features = None

	def __init__(self, m): 
		self.hook = m.register_forward_hook(self.hook_fn)
	
	def hook_fn(self, module, linput, output): 
		self.features = np.squeeze(output.data.cpu().numpy())
		
	def remove(self): 
		self.hook.remove()


class CAM():
	def __init__(self, model, device, last_conv_layer='layer3', fc_layer_name='fc1'):
		self.device = device
		self.last_conv_layer = last_conv_layer
		self.fc_layer_name = fc_layer_name
		self.model = model


	def run(self, instance, label_instance=None):
		cam, pred_probabilities, label_pred =  self.__get_CAM_class(np.array(instance))
		if (label_instance is not None) and (label_pred != label_instance):
			return None
			#Verbose
			#print("[WARNING] expected classification as class {} but got class {}".format(label_instance,label_pred))
			#print("[WARNING] The Class activation map is for class {}".format(label_instance,label_pred))
		return cam, pred_probabilities


	def __getCAM(self, feature_conv, weight_fc, class_idx):
		cam = weight_fc[class_idx].dot(feature_conv)
		return cam


	def __get_CAM_class(self, instance, multiple=True):
		# Input to tensor
		original_dim = len(instance)
		original_length = len(instance[0])
		instance_to_try = Variable(
			torch.tensor(instance.reshape((1, original_dim, original_length))).float().to(self.device),
			requires_grad=True
		)

		# Hook the last conv layer to keep values
		activated_features = SaveFeatures(self.last_conv_layer)

		# Forward pass
		with torch.no_grad():
			prediction = self.model(instance_to_try)
			pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
		
		# Remove the hook after the pass
		activated_features.remove()

		# Get the weights of the fully connected layer
		weight_softmax = self.fc_layer_name.weight.data.cpu().numpy()

		# Get the predicted class index
		class_idx = topk(pred_probabilities, 1)[1].item()

		# Get the CAM 
		overlay = np.array(
			[self.__getCAM(activated_features.features, weight_softmax, i) for i in range(len(pred_probabilities))]
		) if multiple else self.__getCAM(activated_features.features, weight_softmax, class_idx)
		
		return overlay, pred_probabilities, class_idx
