import numpy as np
from torch.utils.data import Sampler

class BalancedClassBatchSampler(Sampler):
    """
    A custom sampler that ensures balanced sampling from different classes in each epoch.
    It generates a list of indices for a batch.
    """
    def __init__(self, labels, batch_size):
        """
        Initialize the sampler.
        Args:
        - labels (torch.Tensor or np.array): Labels for each sample in the dataset.
        - batch_size (int): Size of each batch.
        """
        self.labels = labels
        self.batch_size = batch_size
        
        # Get all unique class IDs
        self.classes = sorted(list(set(labels.tolist())))
        self.num_classes = len(self.classes)
        
        # Group sample indices by class, e.g., {0: [0, 5, 12, ...], 1: [1, 8, ...], ...}
        self.indices_by_class = {cls: np.where(labels == cls)[0] for cls in self.classes}
        
        # Calculate the approximate total number of samples per epoch (used for __len__)
        self.num_samples = len(labels)
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        """
        Generate an iterator of batch indices. This is the core of the sampler.
        """
        # Create a pool of available indices for each epoch
        # We use a list of lists, where each sublist represents available indices for a class
        available_indices_by_class = [list(self.indices_by_class[cls]) for cls in self.classes]
        
        # Shuffle the order of samples within each class at the beginning of each epoch
        for indices in available_indices_by_class:
            np.random.shuffle(indices)

        # Start round-robin sampling
        class_counter = 0  # Counter used for round-robin between classes
        batch = []
        
        for _ in range(self.num_samples):
            # Draw one sample from each class in turn
            current_class_idx = class_counter % self.num_classes
            
            # If samples for the current class are exhausted, refill the index pool for that class from scratch
            if not available_indices_by_class[current_class_idx]:
                available_indices_by_class[current_class_idx] = list(self.indices_by_class[self.classes[current_class_idx]])
                np.random.shuffle(available_indices_by_class[current_class_idx])
            
            # Pop one from the available indices of the current class
            sample_index = available_indices_by_class[current_class_idx].pop()
            batch.append(sample_index)
            
            # If the batch is full, yield this batch and reset
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
            class_counter += 1

    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches