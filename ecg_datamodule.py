import os
from typing import Optional, Sequence
from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper


class ECGDataModule(LightningDataModule):
    def __init__(
            self,
            config,
            transformations_str,
            t_params,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.config = config
        self.transformations_str = transformations_str
        self.t_params = t_params
        self.num_workers = self.config['dataset']['num_workers']
        self.batch_size = self.config['batch_size']
        self.data_dir = config['dataset']['data_path']
        self.seed = seed
        self.set_params()

    def set_params(self):
        dataset = SimCLRDataSetWrapper(
            self.config['batch_size'], **self.config['dataset'], transformations=self.transformations_str, t_params=self.t_params)
        train_loader, valid_loader = dataset.get_data_loaders() 
        self.num_samples = dataset.train_ds_size
        self.transformations = dataset.transformations
    
    @property
    def num_classes(self):
        return 3

    def prepare_data(self):
        pass

    def train_dataloader(self):
        dataset = SimCLRDataSetWrapper(
            self.config['batch_size'], **self.config['dataset'], transformations=self.transformations_str, t_params=self.t_params)
        train_loader, _ = dataset.get_data_loaders()
        # print('train batchsize:', dataset.batch_size)
        
        return train_loader

    def val_dataloader(self):
        # pretraining val_loader
        dataset = SimCLRDataSetWrapper(
            self.config['eval_batch_size'], **self.config['dataset'], transformations=self.transformations_str, t_params=self.t_params)
        _, valid_loader_self = dataset.get_data_loaders()
        
        # fine-tune val_loader and test_loader without contrastive transforms and with labels
        dataset = SimCLRDataSetWrapper(
            self.config['eval_batch_size'], **self.config['dataset'], transformations=self.transformations_str, 
            t_params=self.t_params, mode="linear_evaluation")
        train_loader_sup, valid_loader_sup = dataset.get_data_loaders()
        # return pre-train validloader, and fine-tuning trainloader and validloader
        return [valid_loader_self, train_loader_sup, valid_loader_sup]

    def test_dataloader(self):
        dataset = SimCLRDataSetWrapper(
            self.config['eval_batch_size'], **self.config['dataset'], transformations=self.transformations_str, 
            t_params=self.t_params, mode="linear_evaluation", test=True)
        _, test_loader_sup = dataset.get_data_loaders()
        return test_loader_sup

    def default_transforms(self):
        pass
