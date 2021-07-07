import os
import argparse
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from vonenet import get_model
from src_data import prepare_data
from src_model import VOneResnet_FPN_Volumetric, VOneResnet_FPN, Response

import torch
import torch.nn as nn
from torch.optim import Adam

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"
ROIbeta_root = nsd_root + "roibeta/"

def get_args():
	parser = argparse.ArgumentParser(description='Train the whole brain voxel-wise VoneResnet50_FPN_response encoding model.')
	parser.add_argument('--subject', type=int, help='subject ID, range from 1 to 8')
	parser.add_argument('--nepochs', type=int, help='total training epochs')
	parser.add_argument('--lr', type=float, help='learning rate')
	parser.add_argument('--batchsize', type=int, help='batch size for training')

	args = parser.parse_args()

	return args

def compute_acc(prediction, observation, voxel_mask, metric='pearson'):
	accuracy = []
	voxel_mask = voxel_mask.cpu().detach().numpy().flatten().astype(bool)
	for i in range(prediction.shape[0]):
		pred = prediction[i].cpu().detach().numpy().flatten()
		pred = pred[voxel_mask.flatten()].astype(np.float32)

		true = observation[i].cpu().detach().numpy().flatten()
		true = true[voxel_mask.flatten()].astype(np.float32)

		if metric == 'pearson':
			acc = pearsonr(pred, true)[0]
			accuracy.append(acc)
	return np.mean(accuracy)

def train(model, train_data, voxel_mask, criterion, optimizer, device):
	model.train()
	trn_loss = []
	trn_acc = []  
	voxel_mask = voxel_mask.cuda()
	for batch_idx, data in enumerate(train_data):
		stimuli, target = data['image'].to(device), data['label'].to(device)
		optimizer.zero_grad()
		output = model(stimuli)
		output = output*voxel_mask # mask the output
		loss = criterion(output, target)
		trn_loss.append(loss.item())

		loss.backward()
		optimizer.step()

		trn_acc.append(compute_acc(output,target,voxel_mask))
		
	return np.mean(trn_loss), np.mean(trn_acc)

def validate(model, val_data, voxel_mask, criterion, device):
	model.eval()
	val_loss = []
	val_acc = []
	voxel_mask = voxel_mask.cuda()
	for batch_idx, data in enumerate(val_data):
		stimuli, target = data['image'].to(device), data['label'].to(device)
		output = model(stimuli)
		output = output*voxel_mask # mask the output
		loss = criterion(output, target)
		val_loss.append(loss.item())
		val_acc.append(compute_acc(output,target,voxel_mask))

	return np.mean(val_loss), np.mean(val_acc)

def main():
	args = get_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device) 
	output_dir = './VoneResnet50FPN_weights/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	voxel_mask_full = np.asanyarray(nib.load(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%args.subject).dataobj)
	voxel_mask  = np.nan_to_num(voxel_mask_full)
	voxel_mask = torch.from_numpy(voxel_mask).float()

	trainset, valset = prepare_data(subject=args.subject, voxelmask=True, batchsize=args.batchsize)

	base_model = get_model(model_arch='resnet50', pretrained=True)
	model = VOneResnet_FPN_Volumetric(VOneResnet_FPN(base_model), Response())
	model.to(device)
	print(model)

	for param in model.img2feature.base_model.parameters():
		param.requires_grad = False
	criterion = nn.MSELoss()
	parameters = []
	for child in list(model.img2feature.children())[-7:]:
		parameters.extend(child.parameters())        
	parameters.extend(model.feature2response.parameters())
	optimizer = Adam(parameters, lr=args.lr)

	nepochs = args.nepochs
	train_loss = np.zeros(nepochs)
	train_accuracy = np.zeros(nepochs)
	validation_loss = np.zeros(nepochs)
	validation_accuracy = np.zeros(nepochs)
	for epoch in range(nepochs):
		print("==========EPOCH %d=========="%epoch)
		train_loss[epoch], train_accuracy[epoch] = train(model=model, train_data=trainset, voxel_mask=voxel_mask, criterion=criterion, optimizer=optimizer, device=device)
		validation_loss[epoch], validation_accuracy[epoch] = validate(model=model, val_data=valset, voxel_mask=voxel_mask, criterion=criterion, device=device)
		print("Train loss: %4f"%train_loss[epoch] + "; Val loss: %4f"%validation_loss[epoch])
		if (epoch + 1) % 10 == 0:
			path = output_dir + 'model_' + str(epoch+1) + '.pth.tar'
			state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
			torch.save(state, path)

	np.save(output_dir + 'trainloss_epoch%d'%nepochs + '.npy', train_loss)
	np.save(output_dir + 'trainacc_epoch%d'%nepochs + '.npy', train_accuracy)
	np.save(output_dir + 'valloss_epoch%d'%nepochs + '.npy', validation_loss)
	np.save(output_dir + 'valacc_epoch%d'%nepochs + '.npy', validation_accuracy)

if __name__ == '__main__':
	main()