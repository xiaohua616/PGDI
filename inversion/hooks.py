import torch
import torch.nn as nn
import torch.nn.functional as F

def register_hooks(modules):
    hooks = []
    for m in modules:
        # 如果 FeatureHook 是通用的，并且 InstanceMeanHook 是其特例，可以考虑继承
        # 但这里我们直接修改 InstanceMeanHook
        hooks.append(FeatureHook(m)) # 这部分保持不变，除非您也想修改 FeatureHook
    return hooks

class InstanceMeanHook(object):
    def __init__(self, module, use_spatial_mean=True, hook_target='input'):
        """
        初始化钩子。
        :param module: 要附加钩子的 PyTorch 模块。
        :param use_spatial_mean: 布尔值。如果为 True，并且目标数据是4D (N,C,H,W)，
                                 则计算空间维度上的均值得到 (N,C) 的特征。
                                 如果为 False，则存储原始特征（可能是4D或其它维度）。
        :param hook_target: 字符串，'input' 或 'output'。指定是捕获模块的输入 (input[0]) 还是输出。
        """
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        
        # 用于向后兼容，尽量维持原始 self.instance_mean 的行为
        self.instance_mean = None 
        
        # _captured_feature 将根据配置存储特征，供 get_feature() 使用
        self._captured_feature = None
        
        # 存储配置参数
        self._config_use_spatial_mean = use_spatial_mean
        self._config_hook_target = hook_target.lower() # 转小写以防大小写问题

        if self._config_hook_target not in ['input', 'output']:
            raise ValueError("hook_target 必须是 'input' 或 'output'")

    def hook_fn(self, module, input_tensors, output_tensor):
        # 1. 维持 self.instance_mean 的原始行为 (针对 input[0] 的空间均值)
        # 这是为了向后兼容那些可能直接访问 hook.instance_mean 的代码
        if input_tensors and isinstance(input_tensors[0], torch.Tensor):
            original_input_data = input_tensors[0]
            if original_input_data.ndim == 4: # (N,C,H,W)
                self.instance_mean = torch.mean(original_input_data, dim=[2, 3], keepdim=False).clone().detach()
            elif original_input_data.ndim == 2: # (N,C)
                 self.instance_mean = original_input_data.clone().detach()
            else:
                # 如果输入不是预期的2D或4D，旧代码可能也会出问题，这里设为None
                self.instance_mean = None
        else:
            self.instance_mean = None

        # 2. 根据配置处理特征，并存储到 _captured_feature 供 get_feature() 使用
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
                elif feature_val.ndim == 2: # 已经是 (N,C) 或 (N,D)
                    self._captured_feature = feature_val
                else:
                    # 对于其他维度，如果要求空间均值但不可行，可以选择存储原始值或报错/警告
                    # print(f"警告 (InstanceMeanHook for get_feature): use_spatial_mean=True 但目标数据维度为 {feature_val.ndim}。将存储原始特征。")
                    self._captured_feature = feature_val 
            else: # 不进行空间平均，存储原始（或已处理过的）特征
                self._captured_feature = feature_val
        else:
            self._captured_feature = None

    def get_feature(self):
        """
        获取根据初始化配置（hook_target, use_spatial_mean）处理后的特征。
        供 FastMetaSynthesizerWithFewshot 等新代码调用。
        """
        return self._captured_feature

    def clear(self):
        """清除存储的特征。"""
        self.instance_mean = None
        self._captured_feature = None
        
    def remove(self): # 在您的原始代码中是 remove
        self.hook.remove()
        self.clear() # 移除钩子时也清除数据是个好习惯

    def __repr__(self):
        return (f"<InstanceMeanHook on '{self._config_hook_target}' of {self.module.__class__.__name__}, "
                f"use_spatial_mean={self._config_use_spatial_mean}>")


class FeatureHook(object): # 保持不变
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.output = None # 初始化属性
        self.input = None  # 初始化属性

    def hook_fn(self, module, input, output):
        self.output = output
        self.input = input[0]
    
    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanHook(object): # 这个类与修改后的 InstanceMeanHook(..., hook_target='input', use_spatial_mean=True) 功能重叠
                              # 您可以考虑是否保留或用 InstanceMeanHook 替代
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.instance_mean = None # 初始化

    def hook_fn(self, module, input, output):
        if input and isinstance(input[0], torch.Tensor) and input[0].ndim >=3 : # 至少需要NCH才能取dim 2,3
            self.instance_mean = torch.mean(input[0], dim=[2, 3]) 
        else:
            self.instance_mean = None


    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module) # 可能应为 FeatureMeanHook


class FeatureMeanVarHook(): # 保持不变
    def __init__(self, module, on_input=True, dim=[0,2,3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim
        self.var = None # 初始化
        self.mean = None # 初始化

    def hook_fn(self, module, input, output):
        if self.on_input:
            if not input or not isinstance(input[0], torch.Tensor): return
            feature = input[0].clone() 
        else:
            if not isinstance(output, torch.Tensor): return
            feature = output.clone()
        
        # 确保 feature 的维度足够进行 var_mean 操作
        valid_dims_for_op = True
        for d in self.dim:
            if d >= feature.ndim:
                valid_dims_for_op = False
                break
        if valid_dims_for_op:
            self.var, self.mean = torch.var_mean( feature, dim=self.dim, unbiased=True, keepdim=False )
        else:
            # print(f"警告 (FeatureMeanVarHook): 特征维度 {feature.ndim} 不足以在指定维度 {self.dim} 上操作。")
            self.var, self.mean = None, None


    def remove(self):
        self.hook.remove()
        # self.output=None # 这个类没有 self.output


class DeepInversionHook(): # 保持不变 (仅格式化和添加初始化属性)
    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None
        self.r_feature = None # 初始化 r_feature

    def hook_fn(self, module, input, output):
        if not input or not isinstance(input[0], torch.Tensor) or input[0].ndim < 4:
            self.r_feature = torch.tensor(0.0, device=module.weight.device if hasattr(module,'weight') else 'cpu') # 安全默认值
            self.tmp_val = (torch.tensor(0.0), torch.tensor(0.0)) # 安全默认值
            return

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        if not (hasattr(module, 'running_var') and hasattr(module, 'running_mean') and \
                module.running_var is not None and module.running_mean is not None) :
            # 如果模块没有 running_var 或 running_mean (例如，不是一个标准的BN层，或者未初始化)
            # 这种情况下原始代码会出错，这里给个默认0损失
            self.r_feature = torch.tensor(0.0, device=var.device)
            self.tmp_val = (mean.data.clone(), var.data.clone()) # 保存当前值以便update_mmt
            return

        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean.data.clone(), var.data.clone()) # 保存 .data.clone() 以防万一

    def update_mmt(self):
        if self.tmp_val is None: return # 如果 hook_fn 未能成功执行
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data.clone(), var.data.clone()) # 存储 .data 的克隆
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data.clone(),
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data.clone() )

    def remove(self):
        self.hook.remove()