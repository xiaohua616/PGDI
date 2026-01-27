# vgg.py (Modified to match the reference VGG structure for checkpoint loading)
import torch
import torch.nn as nn
import torch.nn.functional as F # Keep for potential functional ReLU
from collections import OrderedDict
import math # For weight initialization if needed, though reference uses nn.init

__all__ = [
    'VGG_Checkpoint', 'vgg16_checkpoint_compatible', 'vgg16_graft_checkpoint_compatible'
]

# Flat configurations, similar to the reference 'cfgs'
model_configs_flat = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16-graft': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
}

def _split_config_into_blocks(cfg_flat):
    """
    Splits a flat VGG configuration list into blocks, where each block ends with 'M' (MaxPool).
    Example: [64, 64, 'M', 128, 'M'] -> [[64, 64, 'M'], [128, 'M']]
    """
    blocks = []
    current_block = []
    for item in cfg_flat:
        current_block.append(item)
        if item == 'M':
            blocks.append(current_block)
            current_block = []
    if current_block: # Should not happen if cfg ends with 'M' as typical for VGG features
        blocks.append(current_block)
    return blocks

def _make_features_from_blocks(cfg_blocks_list, batch_norm_in_features=False):
    """
    Creates the feature extractor (self.features) as an nn.Sequential of nn.Sequential blocks.
    This matches the structure that produces keys like 'features.BLOCK_IDX.LAYER_IDX.param'.
    batch_norm_in_features is hardcoded to False based on reference and error logs.
    """
    feature_blocks_dict = OrderedDict()
    in_channels = 3
    
    for block_idx, block_config in enumerate(cfg_blocks_list):
        block_layers_dict = OrderedDict()
        layer_order_in_block = 0
        for item_in_block in block_config:
            if item_in_block == 'M':
                block_layers_dict[str(layer_order_in_block)] = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_order_in_block += 1
            else: # It's a conv layer channel size
                conv2d = nn.Conv2d(in_channels, item_in_block, kernel_size=3, padding=1)
                block_layers_dict[str(layer_order_in_block)] = conv2d
                layer_order_in_block += 1
                # ReLU is added next, matching the Conv, ReLU sequence from reference
                block_layers_dict[str(layer_order_in_block)] = nn.ReLU(inplace=True)
                layer_order_in_block += 1
                in_channels = item_in_block
        feature_blocks_dict[str(block_idx)] = nn.Sequential(block_layers_dict)
        
    return nn.Sequential(feature_blocks_dict)


class VGG_Checkpoint(nn.Module):
    def __init__(self, features_cfg_flat, num_classes=1000, 
                 classifier_hidden_dim=512, init_weights=True):
        super(VGG_Checkpoint, self).__init__()

        # Determine the number of input channels for the classifier
        # This is the number of output channels of the last conv layer in the features_cfg_flat
        classifier_input_channels = 0
        for item in reversed(features_cfg_flat):
            if isinstance(item, int): # The first number from the end is the last conv output
                classifier_input_channels = item
                break
        if classifier_input_channels == 0:
            raise ValueError("Could not determine classifier input channels from features_cfg_flat.")

        # 1. Build features: nn.Sequential(OrderedDict of nn.Sequential(OrderedDict of layers))
        block_configs = _split_config_into_blocks(features_cfg_flat)
        self.features = _make_features_from_blocks(block_configs, batch_norm_in_features=False)

        # 2. Build classifier: nn.Sequential(OrderedDict of Linear, BatchNorm1d, Linear)
        # Keys will be "0", "1", "2"
        self.classifier = nn.Sequential(OrderedDict([
            ('0', nn.Linear(classifier_input_channels, classifier_hidden_dim)),
            ('1', nn.BatchNorm1d(classifier_hidden_dim)),
            # ReLU after BatchNorm1d is common. If it's functional in original forward, this is fine.
            # If it was a module at index "2" and final Linear at "3", this would need adjustment.
            # But error log for classifier keys only goes up to "2" for weighted layers.
            # The reference VGG_Stock classifier is Linear -> BN -> Linear.
            ('2', nn.Linear(classifier_hidden_dim, num_classes))
        ]))

        if init_weights:
            self._initialize_weights()

    def forward(self, x, return_features=False): # <--- 修改点1：添加 return_features 参数
        features_out = self.features(x) # 获取特征提取器的输出
        
        # 展平特征以用于分类器
        flattened_features = torch.flatten(features_out, 1)
        logits = self.classifier(flattened_features)

        if return_features:
            # 当需要返回特征时，我们需要决定返回哪个阶段的特征。
            # 选项1: 返回展平后、进入分类器前的特征
            # return logits, flattened_features
            
            # 选项2: 返回特征提取器最后的输出（在展平之前）
            #         这通常是多维的特征图。
            #         如果下游代码期望的是展平后的特征，那么选项1更合适。
            #         如果下游代码期望的是原始的、可能用于其他目的的特征图，那么这个选项合适。
            #         根据datafree_kd.py中 t_feat 的使用方式，可能需要一个可以拼接的特征向量。
            #         一种常见的做法是，如果 `t_feat` 要与学生模型的特征进行比较，
            #         它们应该具有相似的维度或来自相似的语义层级。
            #         对于VGG，最后一个卷积块的输出（features_out）或展平后的版本（flattened_features）
            #         都是合理的候选。
            #         如果你的 `datafree_kd.py` 中的 `criterion` (用于计算学生和教师输出之间的损失)
            #         期望 `t_feat` 是一个特定的特征，你需要确保这里返回的是正确的。
            #         例如，如果 DAFL 或类似方法中的 `act` 损失作用于 `t_feat`，它可能期望的是
            #         教师模型最终的logits之前的激活值（即 `flattened_features`）。
            #         
            # 为了与原始的 `teacher(images, return_features=True)` 调用兼容，我们返回 (logits, features)
            # 这里的 `features_out` 是最后一个卷积/池化层之后的输出，在flatten之前。
            # 如果 `train` 函数中的 `t_feat` 用于例如激活蒸馏，通常会用展平后的特征。
            # 让我们返回展平后的特征，因为它更常用于后续的蒸馏损失计算。
            return logits, flattened_features # <--- 修改点2：返回 logits 和展平后的特征
        else:
            return logits # <--- 修改点3：只返回 logits


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def vgg16_checkpoint_compatible(num_classes=10, **kwargs):
    """
    VGG16 model compatible with the checkpoint structure.
    Features: No BatchNorm2d.
    Classifier: Linear -> BatchNorm1d -> Linear.
    """
    return VGG_Checkpoint(model_configs_flat['vgg16'], num_classes=num_classes, **kwargs)

def vgg16_graft_checkpoint_compatible(num_classes=10, **kwargs):
    """
    VGG16-Graft model compatible with the checkpoint structure.
    """
    return VGG_Checkpoint(model_configs_flat['vgg16-graft'], num_classes=num_classes, **kwargs)