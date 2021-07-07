import sys
import os
import numpy as np
from scipy.io import loadmat
import h5py
import nibabel as nib
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"
ROIbeta_root = nsd_root + "roibeta/"

exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
stim_file       = stim_root + "nsd_stimuli.hdf5"

exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

def list_files(dir_path):
	fileNames = []
	for f in os.listdir(dir_path):
		if os.path.isfile(dir_path+f):
			fileNames += [dir_path+f,]
	return sorted(fileNames)

def prepare_stimuli(subject):
	subj_trials = [30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500]
	image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
	image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255.
	image_data_set.close()

	print ('Image data size = ', image_data.shape)
	print ('Image data type = ', image_data.dtype)
	print (np.min(image_data[0]), np.max(image_data[0]))

	data_size = subj_trials[subject-1]
	ordering_data = ordering[:data_size]
	shared_mask   = ordering_data<1000  # the first 1000 indices are the shared indices

	stim_data = image_data[ordering_data]  # reduce to only the samples available thus far

	trn_stim_data = stim_data[~shared_mask]
	val_stim_data = stim_data[shared_mask]
	del image_data, stim_data
	return trn_stim_data, val_stim_data

def prepare_voxels(subject, folder_name, voxelmask=True, zscore=True, load_ext='.nii.gz'):
	
	if voxelmask:
		voxel_mask_full = np.asanyarray(nib.load(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%subject).dataobj)
		voxel_mask  = np.nan_to_num(voxel_mask_full)

	betas = []
	for filename in list_files(folder_name):
		filename_no_path = filename.split('/')[-1]

		if 'betas' in filename_no_path and load_ext in filename_no_path:
			print (filename) 
			values =  np.asanyarray(nib.load(filename).dataobj).transpose((3,0,1,2))

			if voxelmask:
				beta = (values*voxel_mask).astype(np.float32) / 300.

			if zscore: 
				mb = np.mean(beta, axis=0, keepdims=True)
				sb = np.std(beta, axis=0, keepdims=True)
				beta = np.nan_to_num((beta - mb) / (sb + 1e-6))
				print ("<beta> = %.3f, <sigma> = %.3f" % (np.mean(mb), np.mean(sb)))    

			betas.append(beta)  
	
	betas = np.concatenate(betas, axis=0)    
	data_size = betas.shape[0]
	ordering_data = ordering[:data_size]
	shared_mask   = ordering_data<1000  # the first 1000 indices are the shared indices

	trn_voxel_data = betas[~shared_mask]
	val_voxel_data = betas[shared_mask]
	del betas
	return trn_voxel_data, val_voxel_data

class NSDataset(Dataset):
	def __init__(self, input_data, label_data, transform=None):
		self.transform = transform
		self.images = input_data
		self.label = label_data
		
	def __len__(self):
		return self.label.shape[0]
	
	def __getitem__(self, idx):
		img = self.images[idx]
		img = img.permute(1,2,0)
		label = self.label[idx]
		
		if self.transform:
			img = self.transform(transforms.ToPILImage()(img))
			
		sample = {"image": img, "label": label}
		return sample

def prepare_data(subject, voxelmask=True, batchsize=1):
	# return train dataset and val dataset

	trn_stim_data, val_stim_data = prepare_stimuli(subject)
	
	beta_subj = beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/" % (subject,)
	#trn_voxel_data, val_voxel_data = prepare_voxels(subject=subject,folder_name=beta_subj, voxelmask=True, zscore=True, load_ext='.nii.gz')
	trn_voxel_data = np.load("./S8_train_voxels.npy")
	val_voxel_data = np.load("./S8_val_voxels.npy")
	if (trn_stim_data.shape[0] != trn_voxel_data.shape[0]) | (val_stim_data.shape[0] != val_voxel_data.shape[0]):
		raise ValueError("Stimuli num doesn't match with voxel num!!!")
	else:
		trn_size = trn_stim_data.shape[0]
		val_size = val_stim_data.shape[0]
		print ("Validation size =", val_size, ", Training size =", trn_size)
	
	preprocess = transforms.Compose([ transforms.Resize((224, 224)),
									# transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	trn_stim_data = torch.from_numpy(trn_stim_data).float()
	val_stim_data = torch.from_numpy(val_stim_data).float()
	trn_voxel_data = torch.from_numpy(trn_voxel_data).float()
	val_voxel_data = torch.from_numpy(val_voxel_data).float()

	transformed_trn_dataset = NSDataset(trn_stim_data, trn_voxel_data, transform=preprocess)
	transformed_val_dataset = NSDataset(val_stim_data, val_voxel_data, transform=preprocess)

	trainloader = DataLoader(transformed_trn_dataset, batch_size=batchsize, shuffle=True)
	valloader = DataLoader(transformed_val_dataset, batch_size=batchsize, shuffle=False)

	return trainloader, valloader
