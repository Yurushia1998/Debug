#!/bin/bash

# Configure the resources required
#SBATCH -p v100                                                # partition (this is the queue your job will be added to)
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -c 4              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=12:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --mem=48GB                                              # specify memory required per node (here set to 16 GB)
#SBATCH -p v100
#SBATCH --gres=gpu:1 (for 1 GPU)

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=a1745254@student.adelaide.edu.au          # Email to which notifications will be sent
set -e
set -x


CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=50000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=300 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_max_sim_pseudo_normal_active_uncertainty_lr_0_easy_fluctuate_50-0.02_1000-0.02_1000"  

#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=10000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=300 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_max_sim_pseudo_normal_active_uncertainty_lr_0_easy-0.03_1000-0.03_1000"  --extra_name="Fix_20k_batch_300"
#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.002 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=5000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=100 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=0 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_max_sim_pseudo_reset_all-1_1-1_1" 
#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=50000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=300 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_-0.02_1000-0.02_1000" 

