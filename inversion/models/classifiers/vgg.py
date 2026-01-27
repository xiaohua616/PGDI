import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = [
    'VGG', 'vgg8', 'vgg8_bn', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg16_half', 'vgg16_half_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.cfg_arch = cfg # Store cfg for reference if needed elsewhere
        self.batch_norm = batch_norm

        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        # Dynamically set the input features for the classifier based on the last conv layer of block4
        self.classifier = nn.Linear(cfg[4][-1], num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        # This method assumes batch_norm is True for blocks to have BN layers.
        # It fetches the last BN layer from blocks 1, 2, 3, and 4.
        # If batch_norm is False, these would be Conv2d layers.
        bn_layers = []
        if self.batch_norm:
            if len(self.block1) > 0 and isinstance(self.block1[-1], nn.BatchNorm2d):
                 bn_layers.append(self.block1[-1])
            if len(self.block2) > 0 and isinstance(self.block2[-1], nn.BatchNorm2d):
                 bn_layers.append(self.block2[-1])
            if len(self.block3) > 0 and isinstance(self.block3[-1], nn.BatchNorm2d):
                 bn_layers.append(self.block3[-1])
            if len(self.block4) > 0 and isinstance(self.block4[-1], nn.BatchNorm2d):
                 bn_layers.append(self.block4[-1])
        return bn_layers


    def forward(self, x, return_features=False):
        h = x.shape[2]
        feats = []

        out_block0 = self.block0(x)
        x = F.relu(out_block0)
        x = self.pool0(x)
        if return_features: feats.append(x) # f0, after pool0

        out_block1 = self.block1(x)
        x = F.relu(out_block1)
        x = self.pool1(x)
        if return_features: feats.append(x) # f1, after pool1

        out_block2 = self.block2(x)
        x = F.relu(out_block2)
        x = self.pool2(x)
        if return_features: feats.append(x) # f2, after pool2
        
        out_block3 = self.block3(x)
        x = F.relu(out_block3)
        # Conditional pooling as in the original code
        if h == 64: # e.g. for 64x64 input, this pool is applied. For 32x32, it might be skipped.
            x = self.pool3(x)
        if return_features: feats.append(x) # f3, after block3 (and pool3 if applied)

        out_block4 = self.block4(x)
        x = F.relu(out_block4)
        if return_features: feats.append(x) # f4, after block4 relu, before final pool4

        x = self.pool4(x)
        features_for_classifier = x.view(x.size(0), -1)
        if return_features: feats.append(features_for_classifier) # f5, features right before classifier

        out = self.classifier(features_for_classifier)

        if return_features:
            return out, feats
        else:
            return out

    @staticmethod
    def _make_layers(cfg_block, batch_norm=False, in_channels=3):
        layers = []
        for v_idx, v in enumerate(cfg_block):
            if v == 'M': # Unlikely to be used with current cfg structure
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        # Remove the last ReLU. The ReLU for the block output is applied in the forward pass.
        if len(layers) > 0 and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]], # VGG11
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]], # VGG13
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]], # VGG16
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]], # VGG19
    'S': [[64], [128], [256], [512], [512]], # VGG8 (example)
    'D_half': [[64, 64], [128, 128], [256, 256, 256], [256, 256, 256], [256, 256, 256]], # vgg16-half
}


def vgg8(**kwargs):
    model = VGG(cfg['S'], batch_norm=False, **kwargs)
    return model


def vgg8_bn(**kwargs):
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    model = VGG(cfg['A'], batch_norm=False, **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(cfg['B'], batch_norm=False, **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(cfg['D'], batch_norm=False, **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model

# Added vgg16-half models
def vgg16_half(**kwargs):
    """VGG 16-layer model with half channels in last two blocks (configuration "D_half")
    """
    model = VGG(cfg['D_half'], batch_norm=False, **kwargs)
    return model


def vgg16_half_bn(**kwargs):
    """VGG 16-layer model with half channels in last two blocks (configuration "D_half") and batch normalization
    """
    model = VGG(cfg['D_half'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(cfg['E'], batch_norm=False, **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    # Test with vgg16-half_bn
    print("Testing vgg16_half_bn:")
    # Assuming input is 32x32 for CIFAR-like data, num_classes=100
    # For 32x32 input, h=32, so pool3 in the VGG forward pass will be skipped.
    x_cifar = torch.randn(2, 3, 32, 32)
    net_half_bn = vgg16_half_bn(num_classes=100)
    
    # The forward method now returns (logit, list_of_features) if return_features=True
    logit_half, feats_half = net_half_bn(x_cifar, return_features=True)

    print("Logit shape:", logit_half.shape)
    print("Number of feature maps returned:", len(feats_half))
    for i, f in enumerate(feats_half):
        print(f"Feature {i} shape: {f.shape}, Min value: {f.min().item():.4f}")
    
    print("\nBN layers before ReLU (from get_bn_before_relu):")
    bn_layers = net_half_bn.get_bn_before_relu()
    if bn_layers:
        for idx, m in enumerate(bn_layers):
            if isinstance(m, nn.BatchNorm2d):
                print(f'BN layer {idx} found: {m}')
            else:
                print(f'Warning: Layer {idx} is not a BatchNorm2d: {m}')
    else:
        print("No BN layers returned by get_bn_before_relu (likely because batch_norm=False or blocks are empty).")
    print("-" * 30)

    # Test with standard vgg16_bn to compare classifier input size
    print("\nTesting vgg16_bn:")
    net_full_bn = vgg16_bn(num_classes=100)
    logit_full, feats_full = net_full_bn(x_cifar, return_features=True)
    print("Logit shape:", logit_full.shape)
    print("Number of feature maps returned:", len(feats_full))
    for i, f in enumerate(feats_full):
        print(f"Feature {i} shape: {f.shape}, Min value: {f.min().item():.4f}")
    
    print("\nClassifier details for vgg16_half_bn:")
    print(net_half_bn.classifier)
    print("Classifier details for vgg16_bn:")
    print(net_full_bn.classifier)

    # Test with 64x64 input to see if pool3 is applied
    print("\nTesting vgg16_half_bn with 64x64 input:")
    x_64 = torch.randn(2, 3, 64, 64)
    logit_half_64, feats_half_64 = net_half_bn(x_64, return_features=True)
    print("Logit shape:", logit_half_64.shape)
    print("Number of feature maps returned:", len(feats_half_64))
    # The shape of feats_half_64[2] (f2, after pool2) will be different than feats_half_64[3] (f3, after block3 and pool3)
    # For 32x32 input, f2 is NxCx4x4, f3 (after block3, no pool3) is NxCx4x4
    # For 64x64 input, f2 is NxCx8x8, f3 (after block3 and pool3) is NxCx4x4
    for i, f in enumerate(feats_half_64):
        print(f"Feature {i} shape: {f.shape}, Min value: {f.min().item():.4f}")