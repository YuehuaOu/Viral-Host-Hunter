import os
import subprocess
import threading

# 定义 GPU 卡号与任务的映射
gpu_tasks = {
    1: [('tail', 'family'), ('tail', 'genus')],
    2: [('tail', 'species'), ('lysin', 'family')],
    3: [('lysin', 'genus'), ('lysin', 'species')]
}

# 输入文件路径的模板
base_dir = '/data1/lzh/Projects/PaperCode/Viral-Host-Hunter/data/multi_taxonomic_levels'

# 输出目录模板
output_dir = './model/multi_taxonomic_levels'

# 文件路径模板
file_paths = {
    'train_protein': '{}/{}/{}/train_protein.fasta',
    'train_dna': '{}/{}/{}/train_dna.fasta',
    'val_protein': '{}/{}/{}/val_protein.fasta',
    'val_dna': '{}/{}/{}/val_dna.fasta'
}


# 定义一个函数来执行每个任务
def run_task(cuda_device, t, l):
    # 根据 type 和 level 替换文件路径
    train_protein = file_paths['train_protein'].format(base_dir, t, l)
    train_dna = file_paths['train_dna'].format(base_dir, t, l)
    val_protein = file_paths['val_protein'].format(base_dir, t, l)
    val_dna = file_paths['val_dna'].format(base_dir, t, l)

    # 输出任务信息
    print(f"Running task {t}-{l} on GPU {cuda_device}...")

    # 构建命令
    command = f"export CUDA_VISIBLE_DEVICES={cuda_device} && python train_multi_taxonomic_levels.py " \
              f"--train_protein {train_protein} " \
              f"--train_dna {train_dna} " \
              f"--val_protein {val_protein} " \
              f"--val_dna {val_dna} " \
              f"--type {t} " \
              f"--level {l} "

    # 打印命令并运行
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running task {t}-{l} on GPU {cuda_device}: {e}")

    # 完成任务后打印
    print(f"Task {t}-{l} on GPU {cuda_device} completed.")

# 使用多线程来同时运行不同 GPU 卡上的任务
threads = []
for cuda_device, tasks in gpu_tasks.items():
    for t, l in tasks:
        # 使用线程按顺序运行每个 GPU 上的任务
        thread = threading.Thread(target=run_task, args=(cuda_device, t, l))
        threads.append(thread)
        thread.start()

# 等待所有任务完成
for thread in threads:
    thread.join()

print("All tasks completed.")