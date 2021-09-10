#!/bin/bash

# Configure the resources required
#SBATCH -p v100                                                # partition (this is the queue your job will be added to)
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -c 4              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=48:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --mem=96GB                                              # specify memory required per node (here set to 16 GB)
#SBATCH -p v100
#SBATCH --gres=gpu:1 (for 1 GPU)

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=a1745254@student.adelaide.edu.au          # Email to which notifications will be sent


CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64__ieg.pth --AL_model=DivideMix  --num_img=50000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine"  --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_mKnn_max_sim_pseudo_normal_active_uncertainty_CL_full_rigMagMin-0.02_1000-0.02_1000"   --update_loss="4_0_4_0" --threshold_relabel=0.6 --threshold_clean=0.6 --scaled_loss=3 --using_new_features="last"

#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.02 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64__ieg.pth --AL_model=DivideMix  --num_img=5000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine"  --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_mKnn_max_sim_pseudo_normal_active_uncertainty_CL_full_rigMagMin-1_100-1_100"   --update_loss="0_4_0_4" --threshold_relabel=0.6 --threshold_clean=0.6 --scaled_loss=3 --using_new_features="last" 


#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=10000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=300 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_max_sim_pseudo_normal_active_uncertainty_lr_0_easy-0.03_1000-0.03_1000"  --extra_name="Fix_20k_batch_300"
#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.002 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=5000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=100 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=0 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_max_sim_pseudo_reset_all-1_1-1_1" 
#CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.006 --checkpoint_path=ieg/checkpoints/ieg --model_path=ieg/pretrained/model_sym_0.8_cifar100_300_64_.pth --AL_model=DivideMix  --num_img=50000 --select_by_gradient=True --diverse_and_balance="only_clean_balance_mix_by_class_real_features_cosine" --num_clean=300 --noise_pretrained=ieg/data_noise_pretrained/cifar100/0.8_sym.json --warmup_iteration=20000 --use_GMM_pseudo_classification=True --update_probe="relabel_knn_most_confidence_-0.02_1000-0.02_1000" 

