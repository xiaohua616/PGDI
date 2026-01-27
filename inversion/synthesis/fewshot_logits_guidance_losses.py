import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class FewShotLogitsGuidanceLoss(nn.Module):
    """
    Computes a guidance loss based on the logits distribution of real few-shot samples.

    This module pre-processes the few-shot dataset *once* upon initialization by passing
    the images through the teacher model and caching the resulting logits statistics 
    (e.g., mean softmax, mean/variance, or the full set of logits) for each class.

    During the forward pass, it compares the incoming synthetic logits against these
    cached statistics based on the specified loss_type.
    """
    def __init__(self,
                 teacher_model,         
                 fewshot_images,        
                 fewshot_labels,        
                 num_classes,           
                 loss_type='hybrid_kl_min_l2',
                 temperature_kl=2.0,    
                 gaussian_epsilon=1e-6, 
                 mmd_bandwidths=None,   
                 fa_num_factors=3,      
                 hybrid_alpha=0.997,     
                 device='cpu'):
        super().__init__()
        self.teacher_model = teacher_model
        self.fewshot_images = fewshot_images
        self.fewshot_labels = fewshot_labels
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.temperature_kl = temperature_kl
        self.gaussian_epsilon = gaussian_epsilon
        self.hybrid_alpha = hybrid_alpha

        if mmd_bandwidths is None: 
            self.mmd_bandwidths_sq_list = [0.1, 1.0, 10.0, 100.0]
        else: 
            self.mmd_bandwidths_sq_list = [b**2 for b in mmd_bandwidths]

        self.fa_num_factors = fa_num_factors
        self.device = device

        self.target_info_by_class = self._preprocess_fewshot_logits()

    def _preprocess_fewshot_logits(self):
        """
        Internal method to iterate through the few-shot data *once* and cache
        the necessary target statistics (logits, softmaxes, moments) for each class.
        """
        target_info = {}
        if self.fewshot_images is None or self.fewshot_labels is None:
            print("Warning (FewShotLogitsGuidanceLoss): Few-shot images/labels not provided.")
            return target_info

        with torch.no_grad():
            for i in range(self.num_classes):
                class_mask = (self.fewshot_labels == i)
                if not class_mask.any():
                    target_info[i] = None
                    continue

                fs_images_class = self.fewshot_images[class_mask].to(self.device)
                if fs_images_class.numel() == 0:
                    target_info[i] = None
                    continue

                if fs_images_class.ndim == 5 and fs_images_class.shape[1] == 1:
                    fs_images_class = fs_images_class.squeeze(1)

                fs_logits_class = self.teacher_model(fs_images_class)

                if self.loss_type == 'kl_softmax':
                    avg_softmax_probs = F.softmax(fs_logits_class / self.temperature_kl, dim=1).mean(dim=0)
                    target_info[i] = avg_softmax_probs.detach()
                elif self.loss_type == 'gauss_diag_nll':
                    mu = fs_logits_class.mean(dim=0)
                    var = fs_logits_class.var(dim=0, unbiased=False)
                    var_reg = torch.clamp(var, min=self.gaussian_epsilon)
                    target_info[i] = {'mean': mu.detach(), 'var': var_reg.detach()}
                elif self.loss_type == 'gauss_iso_nll':
                    mu = fs_logits_class.mean(dim=0)
                    dim_vars = fs_logits_class.var(dim=0, unbiased=False)
                    var_scalar = dim_vars.mean()
                    var_scalar_reg = torch.clamp(var_scalar, min=self.gaussian_epsilon)
                    target_info[i] = {'mean': mu.detach(), 'var_scalar': var_scalar_reg.detach()}
                elif self.loss_type == 'min_l2_logits':
                    target_info[i] = fs_logits_class.detach()
                elif self.loss_type == 'mmd_logits':
                    target_info[i] = fs_logits_class.detach()
                elif self.loss_type == 'hybrid_kl_min_l2':
                    avg_softmax_probs = F.softmax(fs_logits_class / self.temperature_kl, dim=1).mean(dim=0)
                    target_info[i] = {
                        'kl_target': avg_softmax_probs.detach(),
                        'l2_targets': fs_logits_class.detach()
                    }
                elif self.loss_type == 'fa_nll':
                    print(f"Warning (FewShotLogitsGuidanceLoss): Preprocessing for '{self.loss_type}' (Factor Analysis parameter estimation) is not fully implemented in this example.")
                    target_info[i] = None
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")
        return target_info

    def _rbf_kernel_matrix(self, X, Y, bandwidth_sq_list):
        """ Helper to compute an RBF kernel matrix (summed over multiple bandwidths). """
        XX = X.pow(2).sum(dim=1, keepdim=True)
        YY = Y.pow(2).sum(dim=1, keepdim=True).t()
        XY = X @ Y.t()
        dist_sq = torch.relu(XX + YY - 2 * XY)
        kernel_val = torch.zeros_like(dist_sq)
        for bw_sq in bandwidth_sq_list:
            kernel_val += torch.exp(-dist_sq / (bw_sq + 1e-8))
        return kernel_val / len(bandwidth_sq_list)

    def forward(self, synth_logits, synth_targets):
        """
        Computes the logits guidance loss for a batch of synthetic logits.
        It iterates over the unique classes present in synth_targets, retrieves the
        cached statistics for each class, and computes the loss only for the
        corresponding synthetic logits.
        """
        if not self.target_info_by_class:
            return {
                'total_loss': torch.tensor(0.0, device=synth_logits.device),
                'kl_component': torch.tensor(0.0, device=synth_logits.device),
                'l2_component': torch.tensor(0.0, device=synth_logits.device)
            }

        total_loss_val = torch.tensor(0.0, device=synth_logits.device)
        total_kl_component_val = torch.tensor(0.0, device=synth_logits.device)
        total_l2_component_val = torch.tensor(0.0, device=synth_logits.device)
        num_samples_processed = 0

        for class_idx_val in torch.unique(synth_targets).tolist():
            target_class_info = self.target_info_by_class.get(class_idx_val)
            if target_class_info is None:
                continue

            current_batch_mask = (synth_targets == class_idx_val)
            current_synth_logits = synth_logits[current_batch_mask]

            if current_synth_logits.numel() == 0:
                continue
            
            N_c = current_synth_logits.size(0)
            loss_this_class = torch.tensor(0.0, device=synth_logits.device)
            
            loss_kl_unweighted_class = torch.tensor(0.0, device=synth_logits.device)
            loss_l2_unweighted_class = torch.tensor(0.0, device=synth_logits.device)

            if self.loss_type == 'kl_softmax':
            # Compute KL divergence against the cached mean softmax probabilities.
                target_avg_softmax_probs = target_class_info
                log_probs_synth = F.log_softmax(current_synth_logits / self.temperature_kl, dim=1)
                loss_kl_unweighted_class = F.kl_div(log_probs_synth,
                                           target_avg_softmax_probs.unsqueeze(0).expand_as(log_probs_synth),
                                           reduction='batchmean', log_target=False)
                loss_this_class = loss_kl_unweighted_class
            
            elif self.loss_type == 'gauss_diag_nll':
                # Compute negative log-likelihood under the cached diagonal Gaussian.
                params = target_class_info
                mu, var_reg = params['mean'], params['var']
                dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.sqrt(var_reg)), 1)
                loss_this_class = -dist.log_prob(current_synth_logits).mean()

            elif self.loss_type == 'gauss_iso_nll':
                # Compute NLL under the cached isotropic Gaussian.
                params = target_class_info
                mu, var_scalar_reg = params['mean'], params['var_scalar']
                if var_scalar_reg <=0: 
                    loss_this_class = torch.tensor(0.0, device=synth_logits.device)
                else:
                    cov_matrix = torch.eye(mu.size(0), device=mu.device) * var_scalar_reg
                    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)
                    loss_this_class = -dist.log_prob(current_synth_logits).mean()
            
            elif self.loss_type == 'min_l2_logits':
                # For each synth logit, find the L2 distance to its *closest* real logit in the cached set.
                S_c_tensor = target_class_info
                if S_c_tensor.numel() == 0: continue 
                L_s_expanded = current_synth_logits.unsqueeze(1)
                S_c_expanded = S_c_tensor.unsqueeze(0)
                distances_sq = torch.sum((L_s_expanded - S_c_expanded)**2, dim=2)
                min_distances_sq, _ = torch.min(distances_sq, dim=1)
                loss_l2_unweighted_class = min_distances_sq.mean()
                loss_this_class = loss_l2_unweighted_class
                
            elif self.loss_type == 'mmd_logits':
                # Compute MMD^2 loss between synth logits and the cached set of real logits.
                S_c_tensor = target_class_info
                k_fs = S_c_tensor.shape[0]
                if S_c_tensor.numel() == 0: continue

                if N_c > 1 and k_fs > 1 :
                    # Unbiased MMD^2 estimator
                    K_ss = self._rbf_kernel_matrix(current_synth_logits, current_synth_logits, self.mmd_bandwidths_sq_list)
                    K_rr = self._rbf_kernel_matrix(S_c_tensor, S_c_tensor, self.mmd_bandwidths_sq_list)
                    K_sr = self._rbf_kernel_matrix(current_synth_logits, S_c_tensor, self.mmd_bandwidths_sq_list)
                    term1 = (K_ss.sum() - K_ss.trace()) / (N_c * (N_c - 1) + 1e-8) 
                    term2 = (K_rr.sum() - K_rr.trace()) / (k_fs * (k_fs - 1) + 1e-8)
                    term3 = -2 * K_sr.mean()
                    loss_mmd_sq = term1 + term2 + term3
                    loss_this_class = torch.relu(loss_mmd_sq)
                elif N_c > 0 and k_fs > 0: 
                    # Biased MMD^2 estimator (fallback for batch size 1)
                    K_ss = self._rbf_kernel_matrix(current_synth_logits, current_synth_logits, self.mmd_bandwidths_sq_list)
                    K_rr = self._rbf_kernel_matrix(S_c_tensor, S_c_tensor, self.mmd_bandwidths_sq_list)
                    K_sr = self._rbf_kernel_matrix(current_synth_logits, S_c_tensor, self.mmd_bandwidths_sq_list)
                    loss_mmd_sq_biased = K_ss.mean() + K_rr.mean() - 2 * K_sr.mean()
                    loss_this_class = torch.relu(loss_mmd_sq_biased)
                else: 
                    loss_this_class = torch.tensor(0.0, device=synth_logits.device)

            elif self.loss_type == 'hybrid_kl_min_l2': 
                # Compute a weighted sum of kl_softmax and min_l2_logits.
                target_kl_info = target_class_info['kl_target']
                log_probs_synth_kl = F.log_softmax(current_synth_logits / self.temperature_kl, dim=1)
                loss_kl_unweighted_class = F.kl_div(log_probs_synth_kl,
                                   target_kl_info.unsqueeze(0).expand_as(log_probs_synth_kl),
                                   reduction='batchmean', log_target=False)
                
                target_l2_info = target_class_info['l2_targets']
                if target_l2_info.numel() == 0: 
                    loss_l2_unweighted_class = torch.tensor(0.0, device=synth_logits.device)
                else:
                    L_s_expanded_l2 = current_synth_logits.unsqueeze(1)
                    S_c_expanded_l2 = target_l2_info.unsqueeze(0)
                    distances_sq_l2 = torch.sum((L_s_expanded_l2 - S_c_expanded_l2)**2, dim=2)
                    min_distances_sq_l2, _ = torch.min(distances_sq_l2, dim=1)
                    loss_l2_unweighted_class = min_distances_sq_l2.mean()

                loss_this_class = self.hybrid_alpha * loss_kl_unweighted_class + \
                                  (1 - self.hybrid_alpha) * loss_l2_unweighted_class
            
            elif self.loss_type == 'fa_nll':
                loss_this_class = torch.tensor(0.0, device=synth_logits.device)
            
            # Accumulate components for logging (if calculated)
            if self.loss_type == 'kl_softmax' or self.loss_type == 'hybrid_kl_min_l2':
                total_kl_component_val += loss_kl_unweighted_class * N_c
            if self.loss_type == 'min_l2_logits' or self.loss_type == 'hybrid_kl_min_l2':
                total_l2_component_val += loss_l2_unweighted_class * N_c
            
            total_loss_val += loss_this_class * N_c 
            num_samples_processed += N_c

        return_dict = {}
        if num_samples_processed > 0:
            return_dict['total_loss'] = total_loss_val / num_samples_processed
            return_dict['kl_component'] = total_kl_component_val / num_samples_processed
            return_dict['l2_component'] = total_l2_component_val / num_samples_processed
        else:
            return_dict['total_loss'] = torch.tensor(0.0, device=synth_logits.device)
            return_dict['kl_component'] = torch.tensor(0.0, device=synth_logits.device)
            return_dict['l2_component'] = torch.tensor(0.0, device=synth_logits.device)
            
        return return_dict