import os
import wfdb
import torch
from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class TrainDataset(Dataset):
    def __init__(self, data_path, records, window_size, signal_fs, transforms=None, stride=None, mode='random'):
        """
        Train dataset. Samples random windows from complete signal length. Ignoring the class borders.
        Args:
            data_path (str): Path to the folder containing the record files.
            records (list of str): List of train record names.
            window_size (float): Length of each window in seconds.
            sampling_frequency (int): The sampling frequency (samples per second).
            stride (int): Step size between windows (default is window_size for non-overlapping).
            mode (str): Mode for sampling windows, 'random' or 'sequential'.
        """
        self.data_path = data_path
        self.records = records
        self.fs = signal_fs
        self.window_size = window_size
        self.stride = stride or self.window_size
        self.mode = mode
        self.transforms = transforms
        self.windows = []
        
        # load all signals and calculate windows
        for record in records:
            record_path = os.path.join(data_path, record, f"{record}_ECG_new")
            
            # load record for signal length
            r = wfdb.rdrecord(record_path, channels=[0])
            signal_length = r.sig_len
            # generate windows in order within the boundary of the class segment
            # add +2 to include the last valid window start because the annotations ends are inclusive while range() is exclusive at the upper bound
            for start in range(0, signal_length - self.window_size + 2, self.stride):
                self.windows.append((record, start))
        
        self.remaining_windows = self.windows.copy()
        
        # initial window shuffle for training data corresponds to random sampling
        if self.mode == 'random':
            random.shuffle(self.remaining_windows)
            
        # cashed records is a dictionary containing the record name as key and signal as value
        self.cashed_records = {}

    def __len__(self):
        """
        Returns the number of windows.
        """
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a single window.
        Args:
            idx (int): Index of the window.
        Returns:
            torch.Tensor: Signal window and dummy label.
        """
        # If remaining windows are exhausted, reshuffle (if using random mode)
        if len(self.remaining_windows) == 0 and self.mode == 'random':
            self.remaining_windows = self.windows.copy()
            random.shuffle(self.remaining_windows)

        if isinstance(idx, slice):
            batch_windows, batch_labels = zip(*[self.__getitem__(i) for i in range(*idx.indices(len(self.remaining_windows)))])
            return torch.stack(batch_windows), torch.tensor(batch_labels)  # Shape: (batch_size, window_size), (batch_size,)
        elif hasattr(idx, '__iter__'):  # If idx is iterable (list-like / numpy array)
            batch_windows, batch_labels = zip(*[self.__getitem__(i) for i in idx ])
            return torch.stack(batch_windows), torch.tensor(batch_labels)
        

        # get start through index
        record, start = self.remaining_windows[idx]
        # cash the record if not already stored
        if record not in self.cashed_records:
            record_path = os.path.join(self.data_path, record, f"{record}_ECG_new")
            self.cashed_records[record] = wfdb.rdrecord(record_path, channels=[0]).p_signal.flatten()
        
        
        # extract signal from window and flatten to remove singular second dimension 
        full_signal = self.cashed_records[record]
        window = full_signal[start : start + self.window_size]

        sample = (torch.tensor(window, dtype=torch.float32).reshape(-1, 1), torch.tensor(0, dtype=torch.long))
        # apply transforms
        if self.transforms:
            sample = self.transforms(sample)

        # return signals in tensor
        return sample

    
class AnnotatedDataset(Dataset):
    def __init__(self, data_path, records, window_size, sampling_frequency, transforms=None, stride=None, balanced_classes=None, mode=None, onehot_label=False):
        """
        Val/test dataset. Samples through sliding window from only annotated segments. Respects each segment/class border.
        Args:
            data_path (str): Path to the folder containing the record files.
            records (list of str): List of record names.
            ann_dict (dict): Dictionary with annotations for each record (start, end, label).
            window_size (float): Length of each window in seconds.
            sampling_frequency (int): The sampling frequency (samples per second).
            stride (int): Step size between windows (default is window_size for non-overlapping).
        """
        self.data_path = data_path
        self.records = records
        self.window_size = window_size
        self.stride = stride or self.window_size
        self.windows = []
        self.transforms = transforms
        self.mode = mode
        self.balanced_classes = balanced_classes
        self.onehot_label = onehot_label
        
        # create annotation dictionary from record names
        annotation_dict = self._create_annotation_dict(records)
        # down-sample annotation dictionary to 100 Hz (from 1000 Hz)
        down_annotation_dict = self._downsample_ann_dict(annotation_dict, factor=10)

        unique_classes = set()
        # load all signals and calculate windows
        for record in records:
            # loop over the class segments in the annotation dictionary
            for start, end, label in down_annotation_dict[record]:
                if label != 0:
                    # sample windows in order within the boundary of the class segment
                    # add +2 to include the last valid window start because the annotations ends are inclusive while range() is exclusive at the upper bound
                    for s in range(start, end - self.window_size + 2, self.stride):
                        unique_classes.add(label)
                        self.windows.append((record, s, label))
        
        # print('unique_classes', unique_classes)
        self.num_classes = len(unique_classes)
        
        # down-sample majority classes to minority class count when enables
        if self.balanced_classes == True:
            print('Balancing classes in annotated dataset...')
            self._balance_classes()
        
                
        self.remaining_windows = self.windows.copy()
        
        # initial window shuffle for training data corresponds to random sampling
        if self.mode == 'random':
            random.shuffle(self.remaining_windows)
        
        # cashed records is a dictionary containing the record name as key and signal as value
        self.cashed_records = {}

    def _create_annotation_dict(self, records):
        ann_dict = {}
        for record in records:
            # load the annotation file
            file_path = os.path.join(self.data_path, record, f"{record}_ANN.csv")
            annotation_columns = ['start1', 'end1', 'score1', 'start2', 'end2', 'score2', 'start3', 'end3', 'score3', 'start_cons', 'end_cons', 'score_cons']
            df = pd.read_csv(file_path, header=None, names=annotation_columns)
            segments = [
                (int(start), int(end), int(c)) for (start, end, c) in zip(df['start_cons'], df['end_cons'], df['score_cons'])
                if pd.notna(start) and pd.notna(end)
            ]
            ann_dict[record] = segments
        return ann_dict

    def _downsample_ann_dict(self, ann_dict, factor=10):
        down_ann_dict = {}
        for key, seg_list in ann_dict.items():
            # annotation is a tuple (start, end, label) with 1-indexing
            down_anns = []
            for ann in seg_list:
                start, end, label = ann
                new_start = np.ceil(start / factor)
                new_end = np.floor((end / factor))
                down_anns.append((int(new_start), int(new_end), label))
            down_ann_dict[key] = down_anns
        return down_ann_dict

    def _balance_classes(self, seed=42):
        """
        Balances the dataset by down-sampling the larger classes to match the minority class count,
        using pandas for efficient group-by sampling.
        """
        # Convert self.windows into a DataFrame for easier manipulation.
        df = pd.DataFrame(self.windows, columns=['record', 'start', 'label'])
        
        # Print class distribution before balancing.
        print("Before", df['label'].value_counts().sort_index().to_dict())
        
        # Determine the minimum count among the classes.
        min_count = df['label'].value_counts().min()
        
        # For each class, sample min_count rows (using a fixed seed for reproducibility).
        df_balanced = pd.concat([
            group.sample(n=min_count if len(group) == min_count else 5 * min_count, random_state=seed)
            for _, group in df.groupby('label')
        ])
              
        # Update self.windows to the balanced set, converting DataFrame rows back to tuples.
        self.windows = list(df_balanced.itertuples(index=False, name=None))
        
        # Print class distribution after balancing.
        print("After", pd.Series([x[2] for x in self.windows]).value_counts().sort_index().to_dict())
        print()
            
    def __len__(self):
        """
        Returns the number of windows.
        """
        return len(self.windows)
    
    # def __getitem__(self, idx):
    #     """
    #     Get a single window with its label.
    #     Args:
    #         idx (int): Index of the window.
    #     Returns:
    #         torch.Tensor: Signal window and corresponding label.
    #     """
    #     if isinstance(idx, slice):
    #         batch_windows, batch_labels = zip(*[self.__getitem__(i) for i in range(*idx.indices(len(self.remaining_windows)))])
    #         return torch.stack(batch_windows), torch.tensor(batch_labels)  # Shape: (batch_size, window_size), (batch_size,)

    #     # if exhausted, refresh windows
    #     if len(self.remaining_windows) == 0:
    #         self.remaining_windows = self.windows.copy()

    #     # access the item by index instead of popping
    #     record, start, label = self.remaining_windows[idx]
        
    #     # cache records if not already cached
    #     if record not in self.cashed_records:
    #         record_path = os.path.join(self.data_path, record, f"{record}_ECG_new")
    #         self.cashed_records[record] = wfdb.rdrecord(record_path, channels=[0]).p_signal.flatten()

    #     # extract the signal from the window
    #     full_signal = self.cashed_records[record]
    #     window = full_signal[start: start + self.window_size]
        
    #     # apply transforms
    #     if self.transforms:
    #         window = self.transforms(window)
        
    #     # return the signal and label
    #     return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        """
        Get a single window with its label.
        Args:
            idx (int): Index of the window.
        Returns:
            tuple: A sample tuple containing the signal window (of shape (window_size, 1))
                and the corresponding label.
        """
        # handle slice or iterable indices for batching
        if isinstance(idx, slice):
            batch_samples = [self.__getitem__(i) for i in range(*idx.indices(len(self.remaining_windows)))]
            # unzip sample and label and stack them
            batch_windows, batch_labels = zip(*batch_samples)
            return torch.stack(batch_windows), torch.tensor(batch_labels)
        
        # if remaining windows are exhausted, refresh them
        if len(self.remaining_windows) == 0:
            self.remaining_windows = self.windows.copy()

        # get record, start index, and label from the remaining windows
        record, start, label = self.remaining_windows[idx]

        # cache the record if it hasn't been cached yet
        if record not in self.cashed_records:
            record_path = os.path.join(self.data_path, record, f"{record}_ECG_new")
            self.cashed_records[record] = wfdb.rdrecord(record_path, channels=[0]).p_signal.flatten()

        # extract the signal window from the cached record
        full_signal = self.cashed_records[record]
        window = full_signal[start: start + self.window_size]
        window_tensor = torch.tensor(window, dtype=torch.float32).reshape(-1, 1)
        
        if self.onehot_label:
            # go from classes [1,2,3] to [0,1,2] for one-hot encoding
            label = label - 1
            label_tensor = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).float()
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        sample = (window_tensor, label_tensor)

        # apply transforms to the sample if provided
        if self.transforms:
            sample = self.transforms(sample)

        return sample


if __name__ == "__main__":

    # Example usage:
    # Define the path to the dataset and the records to be used for training/validation/testing
    butqdb_path = '/path/to/butqdb/'  # Replace with your actual path
    
    # Set up the dataset and dataloader
    data_path = butqdb_path 
    train_records = ['104001', '105001', '115001', '118001', '121001', '125001', '126001']
    validation_records = ['103001', '103002', '103003', '111001', '113001', '123001']
    test_records = ['100001', '100002', '114001', '122001', '124001']

    record_names = sorted(train_records + validation_records + test_records)
    
    window_size = 2.5  # Length of each window (in seconds)
    signal_fs = 100  # Down-sampled signal of 100 Hz (samples per second)

    stride = None  # Optional, overlapping windows

    # annotation_dict = annotation_store(butqdb_path, record_names, 'consensus')
    # down_annotation_dict = downsample_ann_dict(annotation_dict, factor=10)

    # Create Dataset and DataLoader
    train_dataset = TrainDataset(data_path, train_records, window_size, signal_fs, stride, mode='random')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # validation_dataset = AnnotatedDataset(data_path, validation_records, down_annotation_dict, window_size, signal_fs, stride)
    # validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    
    # test_dataset = AnnotatedDataset(data_path, test_records, down_annotation_dict, window_size, signal_fs, stride)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    # # Iterate through the DataLoader
    # for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
    #     plt.plot(batch[0])
    #     break
    #     #     print(f"Batch {batch_idx + 1}: {batch.shape}", end='\r')