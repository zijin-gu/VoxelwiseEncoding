import torch
import torch.nn as nn
from torch.nn import Conv2d, UpsamplingBilinear2d, AvgPool2d, ConvTranspose3d, Linear, ReLU, Sequential
import torchvision
from vonenet import get_model
import os
from collections import OrderedDict 


 # the feature extractor
class VOneResnet_FPN(nn.Module):
	def __init__(self, basemodel):
		super(VOneResnet_FPN, self).__init__()
		self.base_model = Sequential(basemodel.module.vone_block, basemodel.module.bottleneck, *(list(basemodel.module.model.children())[:-2]))
		self.toplayer  = Conv2d(2048, 256, kernel_size=1, stride=1) #(layer4)
		self.latlayer1 = Conv2d(1024, 256, kernel_size=1, stride=1) #(layer3)
		self.latlayer2 = Conv2d(512, 256, kernel_size=1, stride=1) #(layer2)
		self.latlayer3 = Conv2d(256, 256, kernel_size=1, stride=1) #(layer1)
		
		self.smooth1 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.smooth2 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.smooth3 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.activation = OrderedDict()
        
		self.base_model[2][2].relu.register_forward_hook(self.get_activation('layer1'))
		self.base_model[3][3].relu.register_forward_hook(self.get_activation('layer2'))
		self.base_model[4][5].relu.register_forward_hook(self.get_activation('layer3'))
		self.base_model[5][2].relu.register_forward_hook(self.get_activation('layer4'))  
        
	def get_activation(self,layer_name):
		def hook(module, input, output):
			self.activation[layer_name] = output.detach()
		return hook

	def _upsample_add(self, x, y, h, w):
		out = UpsamplingBilinear2d(size=(h,w))(x)
		return out + y 

	def forward(self, x):
		
		out = self.base_model(x)
		layer1 = self.activation['layer1']
		layer2 = self.activation['layer2']
		layer3 = self.activation['layer3']
		layer4 = self.activation['layer4']
		p5 = self.toplayer(layer4)
		p4 = self._upsample_add(p5, self.latlayer1(layer3), 14, 14)
		p4 = self.smooth1(p4)  
		p3 = self._upsample_add(p4, self.latlayer2(layer2), 28, 28)
		p3 = self.smooth2(p3)
		p2 = self._upsample_add(p3, self.latlayer3(layer1), 56, 56)
		p2 = self.smooth3(p2)
		
		z = torch.cat([AvgPool2d(kernel_size=(56,56))(p2), AvgPool2d(kernel_size=(28,28))(p3), AvgPool2d(kernel_size=(14,14))(p4), AvgPool2d(kernel_size=(7,7))(p5)], dim=1)
		z = torch.squeeze(z,2)
		z = torch.squeeze(z,2)

		return z
    
class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(x.shape[0], self.shape[0], self.shape[1], self.shape[2], self.shape[3])

# ceate the response model
class Response(nn.Module):
	def __init__(self):
		super(Response, self).__init__()
		self.convtrans3d_stack = nn.Sequential(
			Linear(1024, 4*12*4*1024),
			ReLU(),
			Reshape(1024,4,12,4),
			ConvTranspose3d(1024,512,kernel_size=(3,3,3), stride=(2,2,2)),
			ReLU(),
			ConvTranspose3d(512,256,kernel_size=(3,3,3), stride=(2,2,2)),
			ReLU(),
			ConvTranspose3d(256,128,kernel_size=(3,3,3), stride=(2,2,2)),
			ReLU(),
			ConvTranspose3d(128,1,kernel_size=(4,1,2), stride=(2,1,2)),
			#ReLU()
		)
	def forward(self, z):
		y = self.convtrans3d_stack(z)
		y = torch.squeeze(y,1)
		return y

# connect the two models
class VOneResnet_FPN_Volumetric(nn.Module):
	def __init__(self, modelA, modelB):
		super(VOneResnet_FPN_Volumetric, self).__init__()
		self.img2feature = modelA
		self.feature2response = modelB
		
	def forward(self, x):
		x = self.img2feature(x)
		y = self.feature2response(x)
		return y