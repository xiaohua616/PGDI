# fewshot_feature_guidance_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# --- MMD Loss Helper Functions ---

def _rbf_kernel_sum(X: torch.Tensor, Y: torch.Tensor, sigmas: List[float], BATCH_SIZE_HIGH_DIM: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the kernel sum components for MMD (K_XX, K_YY, K_XY) using multiple RBF kernels.
    Uses chunking (batching) to avoid OOM errors on large distance matrices.
    """
    dtype = X.dtype if X.nelement() > 0 else (Y.dtype if Y.nelement() > 0 else torch.float32)
    device = X.device if X.nelement() > 0 else (Y.device if Y.nelement() > 0 else 'cpu')

    K_XX_total = torch.tensor(0.0, dtype=dtype, device=device) 
    K_YY_total = torch.tensor(0.0, dtype=dtype, device=device) 
    K_XY_total = torch.tensor(0.0, dtype=dtype, device=device) 

    m = X.size(0)
    n = Y.size(0)

    if m == 0 or n == 0:
        return K_XX_total, K_YY_total, K_XY_total

    X_sq_norms = torch.sum(X * X, dim=1, keepdim=True)  # (m, 1)
    Y_sq_norms = torch.sum(Y * Y, dim=1, keepdim=True)  # (n, 1)

    # Chunked computation for K_XX
    for i in range(0, m, BATCH_SIZE_HIGH_DIM):
        X_batch_i = X[i:i + BATCH_SIZE_HIGH_DIM]
        X_sq_batch_i = X_sq_norms[i:i + BATCH_SIZE_HIGH_DIM]
        for j in range(0, m, BATCH_SIZE_HIGH_DIM):
            X_batch_j = X[j:j + BATCH_SIZE_HIGH_DIM]
            X_sq_batch_j = X_sq_norms[j:j + BATCH_SIZE_HIGH_DIM]
            dist_sq = torch.cdist(X_batch_i, X_batch_j, p=2).pow(2)
            for sigma_sq in [s**2 for s in sigmas]: 
                K_XX_total += torch.exp(-dist_sq / (2 * sigma_sq)).sum()

    # Chunked computation for K_YY
    for i in range(0, n, BATCH_SIZE_HIGH_DIM):
        Y_batch_i = Y[i:i + BATCH_SIZE_HIGH_DIM]
        Y_sq_batch_i = Y_sq_norms[i:i + BATCH_SIZE_HIGH_DIM]
        for j in range(0, n, BATCH_SIZE_HIGH_DIM):
            Y_batch_j = Y[j:j + BATCH_SIZE_HIGH_DIM]
            Y_sq_batch_j = Y_sq_norms[j:j + BATCH_SIZE_HIGH_DIM]
            dist_sq = torch.cdist(Y_batch_i, Y_batch_j, p=2).pow(2)
            for sigma_sq in [s**2 for s in sigmas]:
                K_YY_total += torch.exp(-dist_sq / (2 * sigma_sq)).sum()
                
    # Chunked computation for K_XY
    for i in range(0, m, BATCH_SIZE_HIGH_DIM):
        X_batch_i = X[i:i + BATCH_SIZE_HIGH_DIM]
        X_sq_batch_i = X_sq_norms[i:i + BATCH_SIZE_HIGH_DIM]
        for j in range(0, n, BATCH_SIZE_HIGH_DIM):
            Y_batch_j = Y[j:j + BATCH_SIZE_HIGH_DIM]
            Y_sq_batch_j = Y_sq_norms[j:j + BATCH_SIZE_HIGH_DIM]
            dist_sq = torch.cdist(X_batch_i, Y_batch_j, p=2).pow(2)
            
            for sigma_sq in [s**2 for s in sigmas]:
                K_XY_total += torch.exp(-dist_sq / (2 * sigma_sq)).sum()
                
    return K_XX_total, K_YY_total, K_XY_total


def mmd_loss_rbf(X: torch.Tensor, Y: torch.Tensor, sigmas: List[float], unbiased: bool = True, BATCH_SIZE_HIGH_DIM: int = 512) -> torch.Tensor:
    """
    Computes the MMD^2 loss between two sets of samples X and Y using multiple RBF kernels.
    """
    m = X.size(0)
    n = Y.size(0)
    
    if m == 0 or n == 0:
        return torch.tensor(0.0, device=X.device if m > 0 else Y.device, dtype=X.dtype if m > 0 else (Y.dtype if n > 0 else torch.float32))

    K_XX_sum, K_YY_sum, K_XY_sum = _rbf_kernel_sum(X, Y, sigmas, BATCH_SIZE_HIGH_DIM)

    # Use the unbiased MMD^2 estimator (removes diagonal elements) if specified and valid.
    if unbiased and m == n and m > 1:
        diag_val = float(len(sigmas)) 
        K_XX_sum_no_diag = K_XX_sum - m * diag_val
        K_YY_sum_no_diag = K_YY_sum - n * diag_val
        
        mmd2 = (K_XX_sum_no_diag / (m * (m - 1)) +
                K_YY_sum_no_diag / (n * (n - 1)) -
                2 * K_XY_sum / (m * n))
    else: 
        mmd2 = (K_XX_sum / (m * m) +
                K_YY_sum / (n * n) -
                2 * K_XY_sum / (m * n))
    return mmd2.clamp(min=0) 


# --- CORAL Loss Helper Function ---
def coral_loss(X: torch.Tensor, Y: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Computes the CORAL loss (covariance alignment) between two sets of samples X and Y.
    """
    m, d = X.size()
    n = Y.size(0)
    if m < 2 or n < 2: 
        return torch.tensor(0.0, device=X.device if m > 0 else Y.device, dtype=X.dtype if m > 0 else (Y.dtype if n > 0 else torch.float32))

    mean_X = X.mean(dim=0, keepdim=True)
    mean_Y = Y.mean(dim=0, keepdim=True)
    X_c = X - mean_X
    Y_c = Y - mean_Y

    cov_X = (X_c.t() @ X_c) / (m - 1 + epsilon) 
    cov_Y = (Y_c.t() @ Y_c) / (n - 1 + epsilon) 

    loss = (cov_X - cov_Y).pow(2).sum() / (4 * d * d + epsilon) 
    return loss


# --- Feature Guidance Loss Class ---
class FeatureGuidanceLoss(nn.Module):
    def __init__(self, loss_type: str = 'l2_min', 
                 mmd_sigmas: Optional[List[float]] = None,
                 ot_p: int = 2, 
                 ot_blur: float = 0.05,
                 ot_scaling: float = 0.1, 
                 coral_epsilon: float = 1e-5,
                 mmd_unbiased: bool = True, 
                 mmd_batch_size_high_dim: int = 256): 
        super().__init__()
        self.loss_type = loss_type.lower()
        self.mmd_batch_size_high_dim = mmd_batch_size_high_dim
        self.mmd_unbiased = mmd_unbiased

        if self.loss_type == 'mmd':
            self.mmd_sigmas = mmd_sigmas if mmd_sigmas is not None else [0.1, 1.0, 5.0, 10.0]
            #print(f"FeatureGuidanceLoss: Using MMD with sigmas {self.mmd_sigmas}, unbiased={self.mmd_unbiased}")
        elif self.loss_type == 'sinkhorn':
            try:
                from geomloss import SamplesLoss
                self.ot_p = ot_p
                self.ot_blur = ot_blur
                self.ot_scaling = ot_scaling
                # geomloss is device-agnostic at instantiation, uses tensor device at call time
                self.sinkhorn_loss_fn = SamplesLoss(
                    loss="sinkhorn", 
                    p=self.ot_p, 
                    blur=self.ot_blur, 
                    scaling=self.ot_scaling, 
                    debias=True, 
                    backend="tensorized" 
                )
                #print(f"FeatureGuidanceLoss: Using Sinkhorn (OT) with p={ot_p}, blur={ot_blur}, scaling={ot_scaling}")
            except ImportError:
                print("Falling back to 'l2_mean' loss.")
                self.loss_type = 'l2_mean' 
        elif self.loss_type == 'coral':
            self.coral_epsilon = coral_epsilon
            print(f"FeatureGuidanceLoss: Using CORAL with epsilon {self.coral_epsilon}")
        elif self.loss_type not in ['l2_min', 'l2_mean']:
            raise ValueError(f"Unknown guidance loss type: {self.loss_type}. Supported: l2_min, l2_mean, mmd, sinkhorn, coral.")
        else: # l2_min or l2_mean (or fallback)
            print(f"FeatureGuidanceLoss: Using {self.loss_type}")


    def forward(self, gen_features: torch.Tensor, target_features_for_class: torch.Tensor) -> torch.Tensor:
        """
        Computes the guidance loss between a batch of generated features and the set of real features for that class.
        """
        if gen_features is None or target_features_for_class is None or \
           gen_features.nelement() == 0 or target_features_for_class.nelement() == 0:
            device = 'cpu'
            dtype = torch.float32
            if gen_features is not None and gen_features.nelement() > 0:
                device = gen_features.device
                dtype = gen_features.dtype
            elif target_features_for_class is not None and target_features_for_class.nelement() > 0 :
                device = target_features_for_class.device
                dtype = target_features_for_class.dtype
            return torch.tensor(0.0, device=device, dtype=dtype)
            
        if self.loss_type == 'l2_min':
            dists = torch.cdist(gen_features, target_features_for_class, p=2) 
            min_dists, _ = torch.min(dists, dim=1) 
            return (min_dists ** 2).mean()
        elif self.loss_type == 'l2_mean':
            mean_target_feature = target_features_for_class.mean(dim=0, keepdim=True) 
            return F.mse_loss(gen_features, mean_target_feature.expand_as(gen_features))
        elif self.loss_type == 'mmd':
            return mmd_loss_rbf(gen_features, 
                                target_features_for_class, 
                                sigmas=self.mmd_sigmas, 
                                unbiased=self.mmd_unbiased,
                                BATCH_SIZE_HIGH_DIM=self.mmd_batch_size_high_dim)
        elif self.loss_type == 'sinkhorn':
            if hasattr(self, 'sinkhorn_loss_fn'):
                return self.sinkhorn_loss_fn(gen_features, target_features_for_class)
            else: # Fallback if geomloss was not imported
                mean_target_feature = target_features_for_class.mean(dim=0, keepdim=True)
                return F.mse_loss(gen_features, mean_target_feature.expand_as(gen_features))
        elif self.loss_type == 'coral':
            if gen_features.size(0) < 2 or target_features_for_class.size(0) < 2:
                mean_target_feature = target_features_for_class.mean(dim=0, keepdim=True)
                return F.mse_loss(gen_features, mean_target_feature.expand_as(gen_features))
            return coral_loss(gen_features, target_features_for_class, epsilon=self.coral_epsilon)
        else: 
            print(f"Error: Encountered unknown guidance loss type: {self.loss_type}. Falling back to l2_mean.")
            mean_target_feature = target_features_for_class.mean(dim=0, keepdim=True)
            return F.mse_loss(gen_features, mean_target_feature.expand_as(gen_features))