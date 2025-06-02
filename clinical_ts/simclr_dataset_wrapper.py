from butqdb_dataloaders import TrainDataset, AnnotatedDataset
from .create_logger import create_logger
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from pathlib import Path
import pdb
try:
    import pickle5 as pickle
except ImportError as e:
    import pickle
import random
import torch
from .timeseries_transformations import GaussianNoise, RandomResizedCrop, ChannelResize, Negation, DynamicTimeWarp, DownSample, TimeWarp, TimeOut, ToTensor, BaselineWander, PowerlineNoise, EMNoise, BaselineShift, TGaussianNoise, TRandomResizedCrop, TChannelResize, TNegation, TDynamicTimeWarp, TDownSample, TTimeOut, TBaselineWander, TPowerlineNoise, TEMNoise, TBaselineShift, TGaussianBlur1d, TNormalize, Transpose

logger = create_logger(__name__)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def transformations_from_strings(transformations, t_params):
    if transformations is None:
        return [ToTensor(), Transpose()]

    def str_to_trafo(trafo):
        if trafo == "RandomResizedCrop":
            return TRandomResizedCrop(crop_ratio_range=t_params["rr_crop_ratio_range"], output_size=t_params["output_size"])
        elif trafo == "ChannelResize":
            return TChannelResize(magnitude_range=t_params["magnitude_range"])
        elif trafo == "Negation":
            return TNegation()
        elif trafo == "DynamicTimeWarp":
            return TDynamicTimeWarp(w=t_params["warps"], r=t_params["radius"])
        elif trafo == "DownSample":
            return TDownSample(downsample_ratio=t_params["downsample_ratio"])
        elif trafo == "TimeWarp":
            return TimeWarp(epsilon=t_params["epsilon"])
        elif trafo == "TimeOut":
            return TTimeOut(crop_ratio_range=t_params["to_crop_ratio_range"])
        elif trafo == "GaussianNoise":
            return TGaussianNoise(var=t_params["gaussian_scale"])
        elif trafo == "BaselineWander":
            return TBaselineWander(C=t_params["bw_c"])
        elif trafo == "PowerlineNoise":
            return TPowerlineNoise(Cmax=t_params["pl_cmax"])
        elif trafo == "EMNoise":
            return TEMNoise(var=t_params["em_var"])
        elif trafo == "BaselineShift":
            return TBaselineShift(Cmax=t_params["bs_cmax"])
        elif trafo == "GaussianBlur":
            return TGaussianBlur1d()
        elif trafo == "Normalize":
            return TNormalize()
        else:
            raise Exception(str(trafo) + " is not a valid transformation")

    # for torch transformations
    trafo_list = [ToTensor(transpose_data=False)] + [str_to_trafo(trafo)
                                                     for trafo in transformations] + [Transpose()]
    return trafo_list


class SimCLRDataSetWrapper(object):

    def __init__(self, batch_size, num_workers, data_path, signal_fs, train_records, val_records, test_records, swav=False, num_crops=7, mode="pretraining", transformations=None, 
                    t_params=None, test=False):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        # butqdb folder path
        self.data_path = Path(data_path)
        self.signal_fs = signal_fs
        
        window_duration = 2.5   # seconds
        self.window_size = int(window_duration * self.signal_fs)

        self.transformations = transformations_from_strings(transformations, t_params)
        self.swav = swav
        self.num_crops = num_crops

        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records
    
        if mode in ["linear_evaluation", "pretraining"]:
            self.mode = mode
        else:
            raise("mode unkown")
        
        # TODO: implement test set handling
        self.test = test
    
    def get_data_loaders(self):
        logger.info("loaded data from " + str(self.data_path))
        data_augment = self._get_simclr_pipeline_transform()
        
        if self.mode == "linear_evaluation":
            # do not use contrastive transforms for linear evaluation
            train_ds, val_ds = self._get_datasets(transforms=data_augment)

        elif self.mode == "pretraining":
            wrapper_transform = SwAVDataTransform(data_augment, num_crops=self.num_crops) if self.swav else SimCLRDataTransform(data_augment)
            train_ds, val_ds = self._get_datasets(transforms=wrapper_transform)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_ds, val_ds)

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader

    def _get_datasets(self, transforms=None):
        if self.mode == "pretraining":
            train_ds = TrainDataset(self.data_path, self.train_records, self.window_size, self.signal_fs, transforms=transforms, stride=None, mode='random')
        elif self.mode == "linear_evaluation":
            # load train data (subset of unsupervised pretraining set) with labels for linear evaluation/fine-tuning
            train_ds = AnnotatedDataset(self.data_path, self.train_records, self.window_size, self.signal_fs, transforms=transforms, stride=None, mode='random', onehot_label=True, balanced_classes=False)
        # if test is true, load the test data instead of the validation data
        if not self.test:
            val_ds = AnnotatedDataset(self.data_path, self.val_records, self.window_size, self.signal_fs, transforms=transforms, stride=None, onehot_label=True, balanced_classes=False)
        else:
            val_ds = AnnotatedDataset(self.data_path, self.test_records, self.window_size, self.signal_fs, transforms=transforms, stride=None, onehot_label=True, balanced_classes=False)
        return train_ds, val_ds

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        data_transforms = transforms.Compose(self.transformations)
        return data_transforms

    def get_train_validation_data_loaders(self, train_ds, val_ds):
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        # print("\n"*5)
        # print('trainloader size:', len(train_loader))
        # print('valloder size:', len(val_loader))
        return train_loader, val_loader

class SimCLRDataTransform(object):
    def __init__(self, transform):
        if transform is None:
            self.transform = lambda x: x
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

class SwAVDataTransform(object):
    def __init__(self, transform, num_crops=7):
        if transform is None:
            self.transform = lambda x: x
        self.transform = transform
        self.num_crops=num_crops

    def __call__(self, sample):
        transformed = [] 
        for _ in range(self.num_crops):
            transformed.append(self.transform(sample)[0])
        return transformed, sample[1]