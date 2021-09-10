# coding=utf-8
"""Define FLAGS of the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


def define_basic_flags():
  """Defines basic flags."""

  flags.DEFINE_integer('max_iteration', 100000, 'Number of iteration')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_float('scaled_loss_clean', 1.0, 'scaled_loss_clean')
  flags.DEFINE_integer('batch_size', 100, 'Batch size')
  flags.DEFINE_integer('val_batch_size', 100, 'Validation data batch size.')
  flags.DEFINE_integer('restore_step', 0, ('Checkpoint id.'
                                           '0: load the latest step ckpt.'
                                           '>0: load the assigned step ckpt.'))
  flags.DEFINE_string('diverse_and_balance', '', 'cluster type for active selection')
  flags.DEFINE_integer('num_clean', 10, 'number of clean image')
  flags.DEFINE_integer('num_background_class', -1, 'number of clean image')
  flags.DEFINE_bool('active', True, 'Using active learning to select clean samples')
  flags.DEFINE_bool('select_by_gradient', True, 'Using DivideMix gradient to select clean samples')
  flags.DEFINE_bool('divide_by_class', False, 'Using DivideMix gradient to select clean samples')
  flags.DEFINE_bool('include_clean', False, 'Using DivideMix gradient to select clean samples')
  flags.DEFINE_bool('pretrained_noise', True, 'Using DivideMix gradient to select clean samples')
  flags.DEFINE_string('model_save_path', 'checkpoints/siamese/', 'Save path of siaseme model')
  flags.DEFINE_string('model_path', 'ieg/pretrained/model_sym_0.8_cifar10_300_64_.pth', 'Save path of siaseme model')
  flags.DEFINE_string('noise_pretrained', 'ieg/data_noise_pretrained/0.8_sym.json', 'noise label of noisy samples')
  flags.DEFINE_string('experiment_name', 'DivideMix_cifar10', 'name of experiment')
  flags.DEFINE_string('AL_model', 'DivideMix', 'name of model for active selection')
  flags.DEFINE_string('extra_name', '', 'name of model')
  flags.DEFINE_integer('num_img', 50000, 'number of img used for active selection')
  flags.DEFINE_integer('warmup_iteration', 0, 'number of warmup iteration using pseudo clean')
  flags.DEFINE_bool('rigged_test', False, 'Use EMA')
  flags.DEFINE_string('using_colab', '', 'Use colab')
  flags.DEFINE_bool('use_GMM_pseudo_classification', False, 'Use GMM for classification')
  flags.DEFINE_string('use_pseudo_label_loss_for_features', 'union_GMM_2_loss', 'Type of loss used for GMM')
  flags.DEFINE_string('update_probe', '', 'update probe or not')
  flags.DEFINE_string('using_loss', '', 'update loss type')
  flags.DEFINE_string('start_proba', '', 'update loss type')
  flags.DEFINE_float('uncer_factor', 1, 'multiplication factor of CL')
  flags.DEFINE_float('scaled_loss', 1.0, 'scaled weight of unsupervised loss')
  flags.DEFINE_string('using_new_features', "", 'using_new_features ')
  flags.DEFINE_string('update_loss', "", 'update_loss')
  flags.DEFINE_string('threshold_relabel',"0.5", 'update probe or not')
  flags.DEFINE_float('threshold_clean',0.5, 'update probe or not')
  flags.DEFINE_string('pretrained_path', '', 'pretrained_path')


  flags.DEFINE_enum('network_name', 'wrn28-10',
                    ['resnet29', 'wrn28-10', 'resnet50'],
                    'Network architecture name')
  flags.DEFINE_string('dataset', 'cifar100_uniform_0.8',
                      'Dataset schema: <dataset>_<noise-type>_<ratio>')
  flags.DEFINE_integer('seed', 12345, 'Seed for selecting validation set')
  flags.DEFINE_enum('method', 'ieg', ['ieg', 'l2r', 'supervised'],
                    'Method to deploy.')
  flags.DEFINE_float('momentum', 0.9,
                     'Use momentum optimizer and the same for meta update')
  flags.DEFINE_string('decay_steps', '500',
                      'Decay steps, format (integer[,<integer>,<integer>]')
  flags.DEFINE_float('decay_rate', 0.1, 'Decay steps')
  flags.DEFINE_integer('eval_freq', 500,
                       'How many steps evaluate and save model')
  flags.DEFINE_string('checkpoint_path', '/tmp/ieg',
                      'Checkpoint saving root folder')
  flags.DEFINE_integer('warmup_epochs', 0, 'Warmup with standard training')
  flags.DEFINE_bool('gcloud', False, 'Use Gcloud')
  flags.DEFINE_string('dst_bucket_project', "aiml-carneiro-research", 'dst_bucket_project')
  flags.DEFINE_string('dst_bucket_name', "aiml-carneiro-research-data", 'dst_bucket_name')
  flags.DEFINE_enum('lr_schedule', 'cosine',
                    ['cosine', 'custom_step', 'cosine_warmup', 'exponential'],
                    'Learning rate schedule.')
  flags.DEFINE_float('cos_t_mul', 1.5, 't_mul of cosine learning rate')
  flags.DEFINE_float('cos_m_mul', 0.9, 'm_mul of cosine learning rate')
  flags.DEFINE_bool('use_ema', True, 'Use EMA')

  # Method related arguments
  flags.DEFINE_float('meta_momentum', 0.9, 'Meta momentum.')
  flags.DEFINE_float('meta_stepsize', 0.1, 'Meta learning step size.')
  flags.DEFINE_float('ce_factor', 5,
                     'Weight of cross_entropy loss (p, see paper).')
  flags.DEFINE_float('consistency_factor', 20,
                     'Weight of KL loss (k, see paper)')
  flags.DEFINE_float(
      'probe_dataset_hold_ratio', 0.02,
      'Probe set holdout ratio from the training set (0.02 indicates 1000 '
      'images for CIFAR datasets).'
  )
  flags.DEFINE_float('grad_eps_init', 0.9, 'eps for meta learning init value')
  flags.DEFINE_enum(
      'aug_type', 'autoaug', ['autoaug', 'randaug', 'default'],
      'Fake autoaugmentation type. See dataset_utils/ for more details')
  flags.DEFINE_bool('post_batch_mode_autoaug', True,
                    'If true, apply batch augmentation.')
  flags.DEFINE_enum('mode', 'train', ['train', 'evaluation'],
                    'Train or evaluation mode.')
  flags.DEFINE_bool('use_imagenet_as_eval', False,
                    'Use imagenet as eval when training on webvision while use '
                    'webvision eval when False')
