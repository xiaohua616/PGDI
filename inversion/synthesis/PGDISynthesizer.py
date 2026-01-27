import os
import inversion 
from typing import Generator 
import torch
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F
import random
import shutil
import time 
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as tv_transforms 
from typing import List, Optional, Tuple 
from .base import BaseSynthesis 
from .fewshot_logits_guidance_losses import FewShotLogitsGuidanceLoss
from .fewshot_feature_guidance_losses import FeatureGuidanceLoss
from inversion.hooks import DeepInversionHook, InstanceMeanHook 
from inversion.criterions import jsdiv, get_image_prior_losses, kldiv 
from inversion.utils import ImagePool, DataIter, clip_images 
from kornia import augmentation 
import logging 

def get_fewshot_data(dataset_name, data_root, n_samples_per_class, num_classes, img_size_tuple, normalizer):
    transform = tv_transforms.Compose([
        tv_transforms.Resize(img_size_tuple[-2:]), 
        tv_transforms.ToTensor(),   
        normalizer  
    ])
    
    actual_num_classes_in_dataset = 0 
    if dataset_name.lower() == 'cifar10':
        dataset_cls = CIFAR10
        actual_num_classes_in_dataset = 10
    elif dataset_name.lower() == 'cifar100':
        dataset_cls = CIFAR100
        actual_num_classes_in_dataset = 100
    else:
        print(f"Warning: Loading few-shot data for dataset {dataset_name} is not supported.")
        return None, None

    if num_classes > actual_num_classes_in_dataset:
        print(f"Warning: Requested {num_classes} classes, but {dataset_name} only has {actual_num_classes_in_dataset}. Using {actual_num_classes_in_dataset} classes.")
        num_classes_to_sample = actual_num_classes_in_dataset 
    else:
        num_classes_to_sample = num_classes

    try:
        dataset = dataset_cls(root=data_root, train=True, download=True, transform=transform)
    except Exception as e:
        print(f"Error: Failed to download/load dataset {dataset_name}: {e}")
        return None, None
        
    class_to_indices = {i: [] for i in range(actual_num_classes_in_dataset)}
    for idx, data_point in enumerate(dataset):
        try:
            _, label = data_point 
            if label < actual_num_classes_in_dataset: 
                class_to_indices[label].append(idx)
        except Exception as e: 
            print(f"Warning: Error processing dataset sample {idx}: {e}")
            continue

    selected_indices = [] 
    for cls_idx in range(num_classes_to_sample):
        if not class_to_indices[cls_idx]: 
            print(f"Warning: Class {cls_idx} has no samples in the dataset.")
            continue
        if len(class_to_indices[cls_idx]) < n_samples_per_class:
            print(f"Warning: Class {cls_idx} only has {len(class_to_indices[cls_idx])} samples, but {n_samples_per_class} were requested. Using all available.")
            selected_indices.extend(class_to_indices[cls_idx])
        else: 
            selected_indices.extend(random.sample(class_to_indices[cls_idx], n_samples_per_class))
    
    if not selected_indices: 
        print("Warning: Failed to select any few-shot samples.")
        return None, None

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=len(selected_indices), shuffle=False) 
    
    try:
        images, labels = next(iter(loader)) 
        return images, labels
    except StopIteration: 
        print("Warning: Failed to retrieve data from few-shot DataLoader (StopIteration).")
        return None, None

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None: 
            p.grad = torch.zeros_like(p, device=p.device) 
        diff = p.data - tar_p.data
        if p.grad.data.device != diff.device: 
            diff = diff.to(p.grad.data.device)
        p.grad.data.add_(diff, alpha=67) 

def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None: 
            p.grad = torch.zeros_like(p, device=p.device)
        if hasattr(tar_p, 'grad') and tar_p.grad is not None: 
            grad_data_to_add = tar_p.grad.data 
            if p.grad.data.device != grad_data_to_add.device: 
                grad_data_to_add = grad_data_to_add.to(p.grad.data.device)
            p.grad.data.add_(grad_data_to_add) 

def reset_l0(model): 
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0": 
            if hasattr(m, 'weight') and m.weight is not None: 
                nn.init.normal_(m.weight, 0.0, 0.02) 
            if hasattr(m, 'bias') and m.bias is not None: 
                nn.init.constant_(m.bias, 0) 

def reset_bn(model): 
    for m in model.modules(): 
        if isinstance(m, (nn.BatchNorm2d)): 
            if hasattr(m, 'weight') and m.weight is not None: 
                nn.init.normal_(m.weight, 1.0, 0.02) 
            if hasattr(m, 'bias') and m.bias is not None: 
                nn.init.constant_(m.bias, 0) 


class PGDISynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 init_dataset=None, iterations=100, lr_g=0.1, 
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=1, oh=1, 
                 save_dir='run/fast_meta_fewshot', transform=None, autocast=None, use_fp16=False, 
                 normalizer=None, device='cpu', distributed=False, lr_z = 0.01, 
                 warmup=10, reset_l0_ep=0, reset_bn_ep=0, bn_mmt=0, 
                 is_maml=1, 
                 dataset_name_for_fewshot='cifar100', 
                 data_root_for_fewshot='./data', 
                 fewshot_n_per_class=5, 
                 guidance_loss_w=0.1, 
                 guidance_loss_type='l2_min', 
                 teacher_feature_source='bn_hooks', 
                 logits_guidance_w=0.0, 
                 logits_loss_type='kl_softmax', 
                 temperature_lg=2.0, 
                 mmd_lg_bandwidths=None, 
                 fa_lg_num_factors=3, 
                 guidance_mmd_sigmas: Optional[List[float]] = None, 
                 guidance_mmd_unbiased: bool = True, 
                 guidance_ot_p: int = 2, 
                 guidance_ot_blur: float = 0.05, 
                 guidance_ot_scaling: float = 0.1, 
                 guidance_coral_epsilon: float = 1e-5,
                 guidance_mmd_calc_batch_size: int = 256, 
                 ):
        super(PGDISynthesizer, self).__init__(teacher, student) 
        self.save_dir = save_dir
        self.img_size_tuple = img_size 
        self.iterations = iterations   
        self.lr_g = lr_g   
        self.lr_z = lr_z   
        self.nz = nz   
        self.adv = adv   
        self.bn = bn   
        self.oh = oh   
        self.bn_mmt = bn_mmt 
        self.ismaml = is_maml 

        self.num_classes = num_classes
        self.distributed = distributed 
        self.synthesis_batch_size = synthesis_batch_size 
        self.sample_batch_size = sample_batch_size     
        self.init_dataset = init_dataset 
        self.use_fp16 = use_fp16   
        self.fewshot_n_per_class=fewshot_n_per_class
        
        if autocast is None: 
            if torch.cuda.is_available() and self.use_fp16: 
                self.autocast = lambda: torch.amp.autocast(device_type='cuda', enabled=self.use_fp16)
            else: 
                self.autocast = inversion.utils.dummy_ctx 
        else: 
            self.autocast = autocast

        self.normalizer = normalizer 
        self.data_pool = ImagePool(root=self.save_dir) 
        self.transform = transform if transform is not None else tv_transforms.ToTensor() 
        self.data_iter = None 

        if isinstance(device, int): 
            if device >= 0: self.device = torch.device(f'cuda:{device}')
            else: self.device = torch.device('cpu') 
        elif isinstance(device, str): 
            self.device = torch.device(device)
        elif isinstance(device, torch.device): 
            self.device = device
        else: 
            print(f"Warning: Unknown device type {type(device)}, defaulting to CPU.")
            self.device = torch.device('cpu')
        
        self.generator = generator.to(self.device).train() 
        self.teacher_model = teacher.to(self.device).eval()   
        self.student_model = student.to(self.device)     

        self.hooks = [] 
        for m in self.teacher_model.modules(): 
            if isinstance(m, nn.BatchNorm2d): 
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )
        
        self.guidance_loss_w = guidance_loss_w 
        self.fewshot_images = None   
        self.fewshot_labels = None   
        self.fewshot_target_features_by_class = {} 
        
        if self.guidance_loss_w > 0:
            self.guidance_criterion = FeatureGuidanceLoss( 
                loss_type=guidance_loss_type,
                mmd_sigmas=guidance_mmd_sigmas,
                mmd_unbiased=guidance_mmd_unbiased, 
                ot_p=guidance_ot_p,
                ot_blur=guidance_ot_blur,
                ot_scaling=guidance_ot_scaling,
                coral_epsilon=guidance_coral_epsilon,
                mmd_batch_size_high_dim=guidance_mmd_calc_batch_size 
            ).to(self.device)
            
        self.teacher_feature_source = teacher_feature_source 
        self.guidance_feature_hooks = [] 

        if self.guidance_loss_w > 0 and fewshot_n_per_class > 0:
            print(f"Loading {fewshot_n_per_class}-shot data for feature guidance loss...")
            fs_images, fs_labels = get_fewshot_data( 
                dataset_name=dataset_name_for_fewshot, 
                data_root=data_root_for_fewshot,
                n_samples_per_class=fewshot_n_per_class,
                num_classes=self.num_classes,
                img_size_tuple=self.img_size_tuple,
                normalizer=self.normalizer 
            )
            if fs_images is not None and fs_labels is not None: 
                self.fewshot_images = fs_images.to(self.device) 
                self.fewshot_labels = fs_labels.to(self.device)
                
                if self.fewshot_images.ndim == 5 and self.fewshot_images.shape[1] == 1:
                    self.fewshot_images = self.fewshot_images.squeeze(1) 
                elif self.fewshot_images.ndim != 4:
                    print(f"Warning: Few-shot (feature guidance) image tensor dim is {self.fewshot_images.shape}, teacher model may expect 4D input (B,C,H,W).")

                if self.teacher_feature_source == 'bn_hooks':
                    print("Using mean output (InstanceMean) of all teacher BN layers for feature guidance.")
                    for m_bn in self.teacher_model.modules():
                        if isinstance(m_bn, nn.BatchNorm2d):
                            self.guidance_feature_hooks.append(InstanceMeanHook(m_bn, use_spatial_mean=True))
                
                elif self.teacher_feature_source == 'all_relu': 
                    print("Using mean output  of all teacher ReLU layers for feature guidance.")
                    for name, module in self.teacher_model.named_modules():
                        if isinstance(module, nn.ReLU):
                            self.guidance_feature_hooks.append(InstanceMeanHook(module, use_spatial_mean=True))
                            
                elif self.teacher_feature_source == 'all_conv': 
                    print("Using mean output of all teacher Conv2d layers for feature guidance.")
                    for name, module in self.teacher_model.named_modules():
                        if isinstance(module, nn.Conv2d):
                            self.guidance_feature_hooks.append(InstanceMeanHook(module, use_spatial_mean=True))
                else: 
                    found_layer = False
                    for name, layer in self.teacher_model.named_modules():
                        if name == self.teacher_feature_source:
                            self.guidance_feature_hooks.append(InstanceMeanHook(layer, use_spatial_mean=True))
                            found_layer = True
                            print(f"Attached InstanceMeanHook to teacher layer: {name} for feature guidance.")
                            break
                    if not found_layer:
                        print(f"Warning: Specified teacher layer '{self.teacher_feature_source}' not found for feature guidance. Falling back to BN layers.")
                        print("Using mean output (InstanceMean) of all teacher BN layers for feature guidance.")
                        for m_bn in self.teacher_model.modules():
                            if isinstance(m_bn, nn.BatchNorm2d):
                                self.guidance_feature_hooks.append(InstanceMeanHook(m_bn, use_spatial_mean=True))

                if not self.guidance_feature_hooks:
                    print("Warning: Failed to attach any feature guidance hooks. Check teacher_feature_source setting.")         
                
                if self.guidance_feature_hooks and self.fewshot_images is not None:
                    with torch.no_grad(): 
                        all_fs_feats = self._extract_features_from_hooks(self.teacher_model, self.fewshot_images, self.guidance_feature_hooks)
                        if all_fs_feats is not None: 
                            for i in range(self.num_classes):
                                mask = (self.fewshot_labels == i) 
                                if mask.any(): self.fewshot_target_features_by_class[i] = all_fs_feats[mask].detach()
                        else: 
                            print("Warning: Failed to extract features from few-shot data for feature guidance. Feature guidance loss will be disabled.")
                            self.guidance_loss_w = 0 
                else: 
                    print("Warning: No feature hooks set or no few-shot images loaded. Feature guidance loss will be disabled.")
                    self.guidance_loss_w = 0
            else: 
                print("Warning: Failed to load few-shot (feature guidance) data. Feature guidance loss will be disabled.")
                self.guidance_loss_w = 0
        else: 
            if self.guidance_loss_w > 0: 
                print("Feature guidance weight > 0 but fewshot_n_per_class is 0. Feature guidance loss will be disabled.")
            self.guidance_loss_w = 0 
            self.guidance_criterion = None


        self.logits_guidance_w = logits_guidance_w 
        if self.logits_guidance_w > 0 and self.fewshot_images is not None and self.fewshot_labels is not None:
            self.logits_guidance_criterion = FewShotLogitsGuidanceLoss(
                teacher_model=self.teacher_model, 
                fewshot_images=self.fewshot_images, 
                fewshot_labels=self.fewshot_labels, 
                num_classes=self.num_classes,
                loss_type=logits_loss_type,   
                temperature_kl=temperature_lg,   
                mmd_bandwidths=mmd_lg_bandwidths, 
                fa_num_factors=fa_lg_num_factors, 
                device=self.device   
            ).to(self.device) 
            print(f"Few-shot Logits guidance loss enabled, type: {logits_loss_type}, weight: {self.logits_guidance_w}")
        else:
            self.logits_guidance_criterion = None 
            if self.logits_guidance_w > 0 : 
                print("Warning: Logits guidance weight > 0 but few-shot images were not loaded. Logits guidance will be disabled.")
                self.logits_guidance_w = 0 


        self.ep = 0 
        self.ep_start_warmup = warmup 
        self.reset_l0_ep = reset_l0_ep 
        self.reset_bn_ep = reset_bn_ep 
        
        self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations, betas=[0.5, 0.999])
        
        self.aug = tv_transforms.Compose([ 
            augmentation.RandomCrop(size=self.img_size_tuple[-2:], padding=4).to(self.device),
            augmentation.RandomHorizontalFlip(p=0.5).to(self.device),
            self.normalizer, 
        ])
            
    def _extract_features_from_hooks(self, model, inputs, hooks_list):
        if not hooks_list: return None 
        for h in hooks_list: 
            if hasattr(h, 'clear'): h.clear()

        _ = model(inputs) 
        
        collected_features = [] 
        for h in hooks_list:
            feat = h.get_feature() 
            if feat is not None:
                if feat.ndim == 4: 
                    feat = F.adaptive_avg_pool2d(feat, (1,1)).view(feat.size(0), -1) 
                elif feat.ndim > 2 and feat.ndim !=2 : 
                    feat = feat.view(feat.size(0), -1)
                collected_features.append(feat)
        
        if not collected_features: return None 
        return torch.cat(collected_features, dim=1) 


    def synthesize(self, targets=None):
        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            loop_start_time = time.time() 

        self.ep += 1 
        self.student_model.eval() 
        self.teacher_model.eval() 
        best_cost = float('inf')  

        if self.reset_l0_ep > 0 and (self.ep % self.reset_l0_ep == 0):
            print(f"Resetting generator l0 layers at epoch {self.ep}")
            reset_l0(self.generator) 
        if self.reset_bn_ep > 0 and (self.ep % self.reset_bn_ep == 0):
            print(f"Resetting generator BN layers at epoch {self.ep}")
            reset_bn(self.generator) 

        best_inputs = None 
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device, requires_grad=True) 
        
        if targets is None: 
            targets_tensor = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=self.device)
            targets_tensor = targets_tensor.sort()[0] 
        else: 
            if not isinstance(targets, torch.Tensor): 
                targets_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
            else: 
                targets_tensor = targets.clone().detach().to(self.device, dtype=torch.long)
            targets_tensor = targets_tensor.sort()[0]

        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()}, 
            {'params': [z], 'lr': self.lr_z}   
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations): 
            with self.autocast(): 
                inputs = fast_generator(z) 
                inputs_aug = self.aug(inputs) 
                
                t_out = self.teacher_model(inputs_aug) 
                
                loss_bn_val = torch.tensor(0.0, device=self.device)
                loss_oh_val = torch.tensor(0.0, device=self.device)
                loss_adv_val = torch.tensor(0.0, device=self.device)
                loss_guidance_mean_val = torch.tensor(0.0, device=self.device)
                loss_logits_guidance_val = torch.tensor(0.0, device=self.device)
                
                current_loss_bn = sum([h.r_feature for h in self.hooks if h.r_feature is not None and h.r_feature.numel() > 0])
                if isinstance(current_loss_bn, torch.Tensor) and current_loss_bn.numel() > 0:
                    loss_bn_val = current_loss_bn
                
                loss_oh_val = F.cross_entropy(t_out, targets_tensor)
                
                if self.adv > 0 and (self.ep >= self.ep_start_warmup): 
                    s_out = self.student_model(inputs_aug) 
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    kld_elementwise = kldiv(s_out, t_out.detach(), reduction='none').sum(1) 
                    loss_adv_val = -(kld_elementwise * mask).mean() 
                
                total_loss = self.bn * loss_bn_val + \
                             self.oh * loss_oh_val + \
                             self.adv * loss_adv_val

                if self.guidance_loss_w > 0 and self.guidance_criterion is not None and self.fewshot_target_features_by_class:
                    gen_guidance_features = self._extract_features_from_hooks(self.teacher_model, inputs_aug, self.guidance_feature_hooks)
                    if gen_guidance_features is not None:
                        loss_guidance_accumulated = torch.tensor(0.0, device=self.device)
                        num_guidance_losses_calculated = 0
                        for target_cls_idx in torch.unique(targets_tensor): 
                            cls_idx_item = target_cls_idx.item()
                            if cls_idx_item in self.fewshot_target_features_by_class: 
                                current_class_mask_in_batch = (targets_tensor == target_cls_idx) 
                                gen_features_for_current_class = gen_guidance_features[current_class_mask_in_batch]
                                
                                if gen_features_for_current_class.nelement() > 0 : 
                                    target_fs_features = self.fewshot_target_features_by_class[cls_idx_item]
                                    loss_g_class = self.guidance_criterion(gen_features_for_current_class, target_fs_features)
                                    loss_guidance_accumulated += loss_g_class * gen_features_for_current_class.size(0) 
                                    num_guidance_losses_calculated += gen_features_for_current_class.size(0)
                        
                        if num_guidance_losses_calculated > 0 :
                            loss_guidance_mean_val  = loss_guidance_accumulated / num_guidance_losses_calculated
                            total_loss += self.guidance_loss_w * loss_guidance_mean_val 
                
                loss_logits_guidance_total_val = torch.tensor(0.0, device=self.device)
                loss_logits_kl_component_val = torch.tensor(0.0, device=self.device)
                loss_logits_l2_component_val = torch.tensor(0.0, device=self.device)

                if self.logits_guidance_w > 0 and self.logits_guidance_criterion is not None:
                    logits_loss_output_dict = self.logits_guidance_criterion(t_out, targets_tensor) 
                    loss_logits_guidance_total_val = logits_loss_output_dict['total_loss']
                    if 'kl_component' in logits_loss_output_dict:
                        loss_logits_kl_component_val = logits_loss_output_dict['kl_component']
                    if 'l2_component' in logits_loss_output_dict:
                        loss_logits_l2_component_val = logits_loss_output_dict['l2_component']

                total_loss += self.logits_guidance_w * loss_logits_guidance_total_val 

                if total_loss.item() < best_cost or best_inputs is None:
                    best_cost = total_loss.item()
                    best_inputs = inputs.data.clone() 

                optimizer.zero_grad() 
                if self.use_fp16 and hasattr(self, 'scaler') and self.scaler is not None: 
                    self.scaler.scale(total_loss).backward() 
                    if self.ismaml: 
                        if it == 0: self.meta_optimizer.zero_grad() 
                        self.scaler.unscale_(optimizer) 
                        fomaml_grad(self.generator, fast_generator) 
                        if it == (self.iterations - 1): self.meta_optimizer.step() 
                    self.scaler.step(optimizer) 
                    self.scaler.update() 
                else: 
                    total_loss.backward() 
                    if self.ismaml: 
                        if it == 0: self.meta_optimizer.zero_grad()
                        fomaml_grad(self.generator, fast_generator)
                        if it == (self.iterations - 1): self.meta_optimizer.step()
                    optimizer.step() 
            
        if self.bn_mmt != 0:
            for h in self.hooks: 
                if hasattr(h, 'update_mmt'): h.update_mmt()

        if not self.ismaml: 
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator) 
            self.meta_optimizer.step() 

        self.student_model.train() 

        if self.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize() 
            elapsed_time_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed_time_seconds = time.time() - loop_start_time

        if best_inputs is not None:
            self.data_pool.add(best_inputs.cpu()) 
        
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None: 
            init_dst_transform = self.transform if self.transform is not None else tv_transforms.ToTensor()
            init_dst = inversion.utils.UnlabeledImageDataset(self.init_dataset, transform=init_dst_transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        
        train_sampler = None 
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=getattr(self, 'num_workers', 4), 
            pin_memory=getattr(self, 'pin_memory', True), sampler=train_sampler)
        self.data_iter = DataIter(loader) 
        
        return {"synthetic": best_inputs.detach() if best_inputs is not None else None}, elapsed_time_seconds
    
    def sample(self):
        try:
            return self.data_iter.next() 
        except StopIteration: 
            print("Warning: DataIter exhausted during sample(). May need to re-run synthesize() to populate the data pool.")
            return None
    
    def get_fewshot_info_for_feature_guidance(self):
        return self.fewshot_n_per_class, self.fewshot_images, self.fewshot_labels