import subprocess
import os

# 1. 定义要搜索的超参数组合
# 列表中的每个字典代表一次独立的实验运行
# 我们现在将 guidance_loss_w, logits_guidance_w, 和 mixed_distill_stop_epoch 加入搜索空间
search_configs = [
    {
        "alpha_initial": 0.9, "alpha_final_stage": 0.2,
        "real_fewshot_initial": 256, "real_fewshot_final_stage": 8,
        "guidance_loss_w": 100, "logits_guidance_w": 1, "mixed_distill_stop_epoch": 300
    }
]


# 2. 定义在所有实验中保持不变的固定参数
# guidance_loss_w, logits_guidance_w, mixed_distill_stop_epoch 已被移除
fixed_params = {
    "batch_size": 256,
    "lr": 0.2,
    "kd_steps": 400,
    "ep_steps": 400,
    "adv": 1.33,
    "oh": 0.001,
    "balance": 0,
    "gpu": 0,
    "T": 20,
    "bn_mmt": 0.9,
    "warmup": 20,
    "epochs": 1200,
    "dataset": "cifar100",
    "method": "demo4",
    "g_steps": 10,
    "lr_z": 0.015,
    "lr_g": 5e-3,
    "teacher": "resnet34",
    "student": "resnet18",
    "is_maml": 1,
    "reset_l0": 200,
    "reset_bn": 200,
    "logits_loss_type": "hybrid_kl_min_l2",
    "teacher_feature_source": "all_conv",
    "guidance_loss_type": "mmd",
    "bn": 5,
    "synthesis_batch_size": "512",
    "seed": 2,
    "fewshot_n_per_class": 5,
    "real_fewshot_per_step": 128,
    "alpha_schedule": "cosine",
    "real_fewshot_schedule": "cosine",
}


# 3. 循环运行所有实验配置
for i, config in enumerate(search_configs):
    # 为每次实验创建唯一的日志标签和保存目录
    # 日志标签现在会自动包含所有搜索的参数
    log_tag = f"exp{i+1}"
    for key, value in config.items():
        # 使用简短的键名让标签更紧凑
        key_map = {
            "alpha_initial": "ai", "alpha_final_stage": "af",
            "real_fewshot_initial": "fsi", "real_fewshot_final_stage": "fsf",
            "guidance_loss_w": "glw", "logits_guidance_w": "lgw",
            "mixed_distill_stop_epoch": "stop"
        }
        short_key = key_map.get(key, key) # 如果没有简短版，则使用原键名
        log_tag += f"_{short_key}_{value}"
        
    save_dir = f"run/{log_tag}"

    print("="*80)
    print(f"Running Experiment {i+1}/{len(search_configs)}")
    print(f"Log Tag: {log_tag}")
    print(f"Config: {config}")
    print("="*80)
    
    # 基础命令
    command = ["python", "fewshot_kd.py"] # <--- 请确保这是你的主程序文件名
    
    # 添加固定参数
    for key, value in fixed_params.items():
        command.extend([f"--{key}", str(value)])
        
    # 添加当前搜索的参数 (从config字典中获取)
    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
        
    # 添加动态生成的参数
    command.extend(["--log_tag", log_tag])
    command.extend(["--save_dir", save_dir])
    
    # 打印将要执行的完整命令（用于调试）
    # print(f"Executing command: {' '.join(command)}")

    # 运行命令
    subprocess.run(command)

print("\nAll experiments finished.")