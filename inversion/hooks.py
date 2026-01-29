import torch
import torch.nn as nn
import torch.nn.functional as F

def register_hooks(modules):
    hooks = []
    for m in modules:
        hooks.append(FeatureHook(m)) 
    return hooks

class InstanceMeanHook(object):
    def __init__(self, module, use_spatial_mean=True, hook_target='input'):
        """
        Initialize the hook.
        :param module: The PyTorch module to attach the hook to.
        :param use_spatial_mean: Boolean. If True, and the target data is 4D (N,C,H,W),
                                 calculate the mean over spatial dimensions to get (N,C) features.
                                 If False, store the original features (which could be 4D or other dimensions).
        :param hook_target: String, 'input' or 'output'. Specifies whether to capture the module's input (input[0]) or output.
        """
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        
        # For backward compatibility, try to maintain the original behavior of self.instance_mean
        self.instance_mean = None 
        
        # _captured_feature will store features based on configuration, for use by get_feature()
        self._captured_feature = None
        
        # Store configuration parameters
        self._config_use_spatial_mean = use_spatial_mean
        self._config_hook_target = hook_target.lower() # Convert to lowercase to prevent case sensitivity issues

        if self._config_hook_target not in ['input', 'output']:
            raise ValueError("hook_target must be 'input' or 'output'")

    def hook_fn(self, module, input_tensors, output_tensor):  
        if input_tensors and isinstance(input_tensors[0], torch.Tensor):
            original_input_data = input_tensors[0]
            if original_input_data.ndim == 4: # (N,C,H,W)
                self.instance_mean = torch.mean(original_input_data, dim=[2, 3], keepdim=False).clone().detach()
            elif original_input_data.ndim == 2: # (N,C)
                 self.instance_mean = original_input_data.clone().detach()
            else:
                self.instance_mean = None
        else:
            self.instance_mean = None

        # Process features based on configuration and store in _captured_feature for use by get_feature()
        data_to_process = None
        if self._config_hook_target == 'input':
            if input_tensors and isinstance(input_tensors[0], torch.Tensor):
                data_to_process = input_tensors[0]
        elif self._config_hook_target == 'output':
            data_to_process = output_tensor
        
        if data_to_process is not None:
            feature_val = data_to_process.clone().detach()
            if self._config_use_spatial_mean:
                if feature_val.ndim == 4: # (N,C,H,W)
                    self._captured_feature = torch.mean(feature_val, dim=[2, 3], keepdim=False)
                elif feature_val.ndim == 2: # Already (N,C) or (N,D)
                    self._captured_feature = feature_val
                else:
                    # For other dimensions, if spatial mean is requested but not feasible, 
                    # choose to store original values or raise error/warning
                    # print(f"Warning (InstanceMeanHook for get_feature): use_spatial_mean=True but target data dimension is {feature_val.ndim}. Storing original features.")
                    self._captured_feature = feature_val 
            else: # Do not perform spatial averaging; store the original (or processed) features
                self._captured_feature = feature_val
        else:
            self._captured_feature = None

    def get_feature(self):
        """
        Get the features processed according to initialization configuration (hook_target, use_spatial_mean).
        For use by code such as FastMetaSynthesizerWithFewshot.
        """
        return self._captured_feature

    def clear(self):
        """Clear stored features."""
        self.instance_mean = None
        self._captured_feature = None
        
    def remove(self): 
        self.hook.remove()
        self.clear() 

    def __repr__(self):
        return (f"<InstanceMeanHook on '{self._config_hook_target}' of {self.module.__class__.__name__}, "
                f"use_spatial_mean={self._config_use_spatial_mean}>")


class FeatureHook(object): 
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.output = None 
        self.input = None  

    def hook_fn(self, module, input, output):
        self.output = output
        self.input = input[0]
    
    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanHook(object): 
                              
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.instance_mean = None 

    def hook_fn(self, module, input, output):
        if input and isinstance(input[0], torch.Tensor) and input[0].ndim >=3 : 
            self.instance_mean = torch.mean(input[0], dim=[2, 3]) 
        else:
            self.instance_mean = None


    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module) 


class FeatureMeanVarHook(): 
    def __init__(self, module, on_input=True, dim=[0,2,3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim
        self.var = None 
        self.mean = None 

    def hook_fn(self, module, input, output):
        if self.on_input:
            if not input or not isinstance(input[0], torch.Tensor): return
            feature = input[0].clone() 
        else:
            if not isinstance(output, torch.Tensor): return
            feature = output.clone()
        
        valid_dims_for_op = True
        for d in self.dim:
            if d >= feature.ndim:
                valid_dims_for_op = False
                break
        if valid_dims_for_op:
            self.var, self.mean = torch.var_mean( feature, dim=self.dim, unbiased=True, keepdim=False )
        else:
            self.var, self.mean = None, None


    def remove(self):
        self.hook.remove()

class DeepInversionHook(): 
    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None
        self.r_feature = None 

    def hook_fn(self, module, input, output):
        if not input or not isinstance(input[0], torch.Tensor) or input[0].ndim < 4:
            self.r_feature = torch.tensor(0.0, device=module.weight.device if hasattr(module,'weight') else 'cpu') 
            self.tmp_val = (torch.tensor(0.0), torch.tensor(0.0)) 
            return

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        if not (hasattr(module, 'running_var') and hasattr(module, 'running_mean') and \
                module.running_var is not None and module.running_mean is not None) :
            self.r_feature = torch.tensor(0.0, device=var.device)
            self.tmp_val = (mean.data.clone(), var.data.clone()) # Save current values for update_mmt
            return

        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean.data.clone(), var.data.clone()) 
    def update_mmt(self):
        if self.tmp_val is None: return 
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data.clone(), var.data.clone()) 
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data.clone(),
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data.clone() )

    def remove(self):
        self.hook.remove()