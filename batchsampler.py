import numpy as np
from torch.utils.data import Sampler

class BalancedClassBatchSampler(Sampler):
    """
    一个自定义的采样器，确保在每个 epoch 中从不同类别中均衡地抽样。
    它会生成一个批次（batch）的索引列表。
    """
    def __init__(self, labels, batch_size):
        """
        初始化采样器。
        参数:
        - labels (torch.Tensor or np.array): 数据集中每个样本的标签。
        - batch_size (int): 每个批次的大小。
        """
        self.labels = labels
        self.batch_size = batch_size
        
        # 获取所有唯一的类别ID
        self.classes = sorted(list(set(labels.tolist())))
        self.num_classes = len(self.classes)
        
        # 按类别对样本索引进行分组，例如: {0: [0, 5, 12, ...], 1: [1, 8, ...], ...}
        self.indices_by_class = {cls: np.where(labels == cls)[0] for cls in self.classes}
        
        # 计算一个 epoch 大约包含的总样本数 (用于 __len__)
        self.num_samples = len(labels)
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        """
        生成一个批次的索引迭代器。这是采样器的核心。
        """
        # 为每个 epoch 创建一个可用的索引池
        # 我们使用一个列表的列表，每个子列表代表一个类别的可用索引
        available_indices_by_class = [list(self.indices_by_class[cls]) for cls in self.classes]
        
        # 在每个 epoch 开始时，打乱每个类别内部的样本顺序
        for indices in available_indices_by_class:
            np.random.shuffle(indices)

        # 开始轮询采样
        class_counter = 0  # 用于在类别间轮询的计数器
        batch = []
        
        for _ in range(self.num_samples):
            # 轮流从每个类别中抽取一个样本
            current_class_idx = class_counter % self.num_classes
            
            # 如果当前类别的样本已经用完，则从头开始重新填充该类别的索引池
            if not available_indices_by_class[current_class_idx]:
                available_indices_by_class[current_class_idx] = list(self.indices_by_class[self.classes[current_class_idx]])
                np.random.shuffle(available_indices_by_class[current_class_idx])
            
            # 从当前类别的可用索引中弹出一个
            sample_index = available_indices_by_class[current_class_idx].pop()
            batch.append(sample_index)
            
            # 如果批次满了，则 yield 这个批次并重置
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
            class_counter += 1

    def __len__(self):
        """返回每个 epoch 的批次数。"""
        return self.num_batches