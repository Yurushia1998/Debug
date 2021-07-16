# coding=utf-8
"""Loader for datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import flags
from ieg.dataset_utils.utils import cifar_process
from ieg.dataset_utils.utils import imagenet_preprocess_image
import numpy as np
import sklearn.metrics as sklearn_metrics
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import keras
from keras import Model,layers
import json
from torchvision import transforms
import torch
import torchvision.models as models
from PIL import Image
from ..models.PreResNet import ResNet18
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import accuracy_score
from google.cloud import storage
from typing import Union
from pathlib import Path
import glob
from tqdm import tqdm
import shutil 
FLAGS = flags.FLAGS
from scipy.cluster.vq import vq
from scipy import stats


def verbose_data(which_set, data, label):
  """Prints the number of data per class for a dataset.

  Args:
    which_set: a str
    data: A numpy 4D array
    label: A numpy array
  """
  text = ['{} size: {}'.format(which_set, data.shape[0])]
  for i in range(label.max() + 1):
    text.append('class{}-{}'.format(i, len(np.where(label == i)[0])))
  text.append('\n')
  text = ' '.join(text)
  tf.logging.info(text)


def shuffle_dataset_original(data, label, others=None, class_balanced=False):
  """Shuffles the dataset with class balancing option.

  Args:
    data: A numpy 4D array.
    label: A numpy array.
    others: Optional array corresponded with data and label.
    class_balanced: If True, after shuffle, data of different classes are
      interleaved and balanced [1,2,3...,1,2,3.].

  Returns:
    Shuffled inputs.
  """
  if class_balanced:
    sorted_ids = []

    for i in range(label.max() + 1):
      tmp_ids = np.where(label == i)[0]
      np.random.shuffle(tmp_ids)
      sorted_ids.append(tmp_ids)

    sorted_ids = np.stack(sorted_ids, 0)
    sorted_ids = np.transpose(sorted_ids, axes=[1, 0])
    ids = np.reshape(sorted_ids, (-1,))

  else:
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)

  if others is None:
    return data[ids], label[ids]
  else:
    return data[ids], label[ids], others[ids]
def shuffle_dataset(data, label, others=None,noise_mask = None, class_balanced=False):
  """Shuffles the dataset with class balancing option.

  Args:
    data: A numpy 4D array.
    label: A numpy array.
    others: Optional array corresponded with data and label.
    class_balanced: If True, after shuffle, data of different classes are
      interleaved and balanced [1,2,3...,1,2,3.].

  Returns:
    Shuffled inputs.
  """
  if class_balanced:
    sorted_ids = []

    for i in range(label.max() + 1):
      tmp_ids = np.where(label == i)[0]
      np.random.shuffle(tmp_ids)
      sorted_ids.append(tmp_ids)

    sorted_ids = np.stack(sorted_ids, 0)
    sorted_ids = np.transpose(sorted_ids, axes=[1, 0])
    ids = np.reshape(sorted_ids, (-1,))

  else:
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)

  if noise_mask is None:
    curr_noise_mask = []
  else:
    curr_noise_mask = noise_mask[ids]
  if others is None and noise_mask is None:
    return data[ids], label[ids]
  elif noise_mask is None:
    return data[ids], label[ids], others[ids]
  else:
    return data[ids], label[ids], others[ids],curr_noise_mask

def get_pretrained_model(name):
    FLAGS = tf.flags.FLAGS
    if name == "resnet34":
        pass
    elif name == "BYOL_torch_100":
        
        resnet = models.resnet50(pretrained=False)
        
        resnet.load_state_dict(torch.load(FLAGS.model_path))

        learner = BYOL(
            resnet,
            image_size = 32,
            hidden_layer = 'avgpool'
        )
        print("Loaded pre-trained model BYOL_torch_100 with success.")
        return learner
    elif name == "BYOL_torch_300":
        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load(FLAGS.model_path))

        learner = BYOL(
            resnet,
            image_size = 32,
            hidden_layer = 'avgpool'
        )
        print("Loaded pre-trained model BYOL_torch_300 with success.")
        return learner
    elif name == "SimCLR2_torch":
        model = ResNetSimCLR(out_dim = 128,base_model = "resnet50")
        state_dict = torch.load(FLAGS.model_path)
        model.load_state_dict(state_dict)
        print("Loaded pre-trained model SimCLR2_torch with success.")
        return model
    elif name == "DivideMix":
        net1 = ResNet18(num_classes=int(FLAGS.dataset.split("_")[0][5:]))
        net2 = ResNet18(num_classes=int(FLAGS.dataset.split("_")[0][5:]))
        weight = torch.load(FLAGS.model_path)
        net1.load_state_dict(weight["net1"])
        net2.load_state_dict(weight["net2"])
        return (net1,net2)

def get_informativeness(prediction_1,prediction_2):
    final_prediction = (prediction_1 + prediction_2)/2
    final_prediction = sorted(final_prediction)
    informativeness = 1 - (final_prediction[-1] - final_prediction[-2])
    return informativeness        

def trainval_split_active_learning(img, label,clean_label,noise_mask, num_val, seed = 0 ,train_index = None):
    """Splits training set and validation set.

    :param img              [ndarray]      All images.
    :param label            [ndarray]      All labels.
    :param num_val          [int]          Number of validation images.
    :param seed             [int]          Random seed for generating the split.

    :return
    """
    FLAGS = tf.flags.FLAGS
    num_img = FLAGS.num_img

    all_features,all_features_1,all_features_2 = [],[],[]
    curr_dataset = FLAGS.dataset.split("_")[0]
    if curr_dataset == "cifar10":
      num_cluster = 10
    elif curr_dataset == "cifar100":
      num_cluster = 100
    elif "mini" in curr_dataset:
      num_cluster = 50
    else:
      num_cluster = 1000

    
    feature_extractor = get_pretrained_model(name = FLAGS.AL_model)
    
    if torch.cuda.is_available():
        if FLAGS.AL_model.startswith("DivideMix"):
            feature_extractor_1 = feature_extractor[0]
            feature_extractor_2 = feature_extractor[1]
  
            feature_extractor_1.eval()
            feature_extractor_2.eval()
            feature_extractor_1.cuda()
            feature_extractor_2.cuda()
            
        else:
            feature_extractor.eval()
            feature_extractor.cuda()
        print("Model to cuda")
    normalize = transforms.Compose([
               transforms.Resize([224,224]),
               transforms.ToTensor(),
               transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ]) 
    if FLAGS.AL_model.startswith("DivideMix"):
      if FLAGS.dataset.startswith("cifar100"):
        normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
        
        
      elif FLAGS.dataset.startswith("cifar10"):
        normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),])
    print("Unique in first "+str(img.shape[0])+": ",np.unique(label[:img.shape[0]],return_counts = True))

    prediction = []
    prediction_label = []
    prediction_proba = []
    all_loss = [[],[]]
    total_img =  FLAGS.num_img
    criterion = torch.nn.CrossEntropyLoss()
    for idx in range(total_img):

        curr_img = img[idx].copy()
        curr_label = label[idx].copy()
        curr_img = Image.fromarray(curr_img, mode='RGB')
        curr_img = normalize(curr_img)
        curr_img = torch.unsqueeze(curr_img,0)
        if torch.cuda.is_available():
            curr_img = curr_img.cuda()
            curr_label = torch.Tensor([curr_label]).long().cuda()
        if FLAGS.select_by_gradient:
            if FLAGS.AL_model.startswith("DivideMix"):
                all_gradient_1 = []
                all_gradient_2 = []
                feature_extractor_1.zero_grad()
                feature_extractor_2.zero_grad()
                all_weight_layer = {}

                prediction_1, curr_feature_1 = feature_extractor_1(curr_img)

                loss = criterion(prediction_1, curr_label)
                all_loss[0].append(loss.clone().detach().cpu().numpy().squeeze())
                loss.backward()
            
                gradient_1 = feature_extractor_1.linear.weight.grad.clone().detach().cpu()
                prediction_1 = nn.functional.softmax(prediction_1)


                prediction_2, curr_feature_2 = feature_extractor_2(curr_img)

                loss_2 = criterion(prediction_2, curr_label)
                all_loss[1].append(loss_2.clone().detach().cpu().numpy().squeeze())
                loss_2.backward()
                gradient_2 = feature_extractor_2.linear.weight.grad.clone().detach().cpu()
                prediction_2 = nn.functional.softmax(prediction_2)


                prediction_1 = prediction_1.detach().cpu().numpy().squeeze()
                prediction_2 = prediction_2.detach().cpu().numpy().squeeze()
               

                curr_feature_1 = curr_feature_1.detach().cpu()
                curr_feature_2 = curr_feature_2.detach().cpu()
                curr_feature = (curr_feature_1.numpy().squeeze() + curr_feature_2.numpy().squeeze()) / 2

                curr_feature_1 = torch.matmul(curr_feature_1, gradient_1.T)
                curr_feature_2 = torch.matmul(curr_feature_2, gradient_2.T)
                curr_feature_1 = curr_feature_1.numpy().squeeze()
                curr_feature_2 = curr_feature_2.numpy().squeeze()

                curr_prediction = (prediction_1 + prediction_2) / 2
                assert abs(np.sum(curr_prediction) - 1.0) < 0.01
                curr_prediction_label = np.argmax(curr_prediction)
                prediction.append(get_informativeness(prediction_1, prediction_2))
                prediction_label.append(curr_prediction_label)
                prediction_proba.append(curr_prediction)


                all_features_1.append(curr_feature_1)
                all_features_2.append(curr_feature_2)
            all_features.append(curr_feature)
        

    print("Finish getting features")
    print("Prediction: ",len(prediction))
    print("Shape of all features: ",np.shape(all_features))
    print("Shape of all features 1: ",np.shape(all_features_1))
    print("Shape of all features 2: ",np.shape(all_features_2))
    prediction = np.array(prediction)
    prediction_label = np.array(prediction_label)
    prediction_proba = np.array(prediction_proba)
    all_features = np.vstack(all_features)
    all_features_1 = np.vstack(all_features_1)
    all_features_2 = np.vstack(all_features_2)
    all_loss = np.array([np.array(all_loss[0]),np.array(all_loss[1])])
    print("Shape feature 1: ",np.shape(all_features))
    print("Shape all loss: ",np.shape(all_loss))
    if FLAGS.AL_model.startswith("DivideMix"):
        feature_extractor_1=feature_extractor_1.cpu()
        feature_extractor_2=feature_extractor_2.cpu()
    else:
        feature_extractor=feature_extractor.cpu()
    del feature_extractor
    torch.cuda.empty_cache()

    noisy_label = label
    noisy_label_array = np.expand_dims(noisy_label.copy(),axis = 1)

    correct_label = np.squeeze(clean_label)
    correct_label_array = np.expand_dims(correct_label.copy(),axis = 1)

    assert img.shape[0] == label.shape[0], 'Images and labels dimension must match.'

    num = img.shape[0]
    trainval_partition = [num - num_val, num_val]
    num_point_per_class = int(FLAGS.num_clean/num_cluster)
    #test_theory(noise_mask,(all_features,all_features_1,all_features_2),correct_label[:FLAGS.num_img],num_class = 10)

    clean_data_idx,all_clean_index = select_representation_test_active(features = (all_features,all_features_1,all_features_2,all_loss),num_point_per_class = num_point_per_class,num_class = num_cluster,correct_label = correct_label,prediction = prediction,prediction_label = prediction_label,prediction_proba = prediction_proba,noise_mask = noise_mask,noisy_label = noisy_label)

    print("Clean id is: ",clean_data_idx)
    
    
   
    if not FLAGS.divide_by_class:
        clean_data_idx = [clean_data_idx]
    clean_data_idx_all = []
    for cluster_clean_idx in clean_data_idx:
        clean_data_idx_all.extend(cluster_clean_idx) 
    if not FLAGS.include_clean:
        print("Separate clean and noisy")
        noisy_training_idx = [i for i in range(FLAGS.num_img) if i not in clean_data_idx_all]
    else:
        print("Include both clean and noisy for noisy")
        noisy_training_idx = [i for i in range(FLAGS.num_img)]
    print("Unique label of clean idx: ")
    print(np.unique(label[clean_data_idx_all],return_counts = True))
    tf.logging.info("Unique label of clean idx: ")
    tf.logging.info(np.unique(label[clean_data_idx_all],return_counts = True))

    print("Unique true label of clean idx: ")
    print(np.unique(clean_label[clean_data_idx_all],return_counts = True))
    tf.logging.info("Unique true label of clean idx: ")
    tf.logging.info(np.unique(clean_label[clean_data_idx_all],return_counts = True))
    print("Noise label of active select: ",label[clean_data_idx_all])
    print("True label of active select: ",clean_label[clean_data_idx_all])
    print("Noise mask label of active select: ",noise_mask[clean_data_idx_all])

    left_clean_index = [index for index in all_clean_index if index not in clean_data_idx_all]
    update_noise_mask = noise_mask[noisy_training_idx]
    return img[noisy_training_idx], label[noisy_training_idx], img[clean_data_idx_all], label[clean_data_idx_all],clean_label[noisy_training_idx],update_noise_mask, img[left_clean_index], label[left_clean_index],all_clean_index,noisy_training_idx,left_clean_index,[all_features,all_features_1,all_features_2]

def load_asymmetric(x, y, noise_ratio, n_val, random_seed=12345):
  """Create asymmetric noisy data."""

  def _generate_asymmetric_noise(y_train, n):
    """Generate cifar10 asymmetric label noise.

    Asymmetric noise confuses
      automobile <- truck
      bird -> airplane
      cat <-> dog
      deer -> horse

    Args:
      y_train: label numpy tensor
      n: noise ratio

    Returns:
      corrupted y_train.
    """
    assert y_train.max() == 10 - 1
    classes = 10
    p = np.eye(classes)

    # automobile <- truck
    p[9, 9], p[9, 1] = 1. - n, n
    # bird -> airplane
    p[2, 2], p[2, 0] = 1. - n, n
    # cat <-> dog
    p[3, 3], p[3, 5] = 1. - n, n
    p[5, 5], p[5, 3] = 1. - n, n
    # automobile -> truck
    p[4, 4], p[4, 7] = 1. - n, n
    tf.logging.info('Asymmetric corruption p:\n {}'.format(p))

    noise_y = y_train.copy()
    r = np.random.RandomState(random_seed)

    for i in range(noise_y.shape[0]):
      c = y_train[i]
      s = r.multinomial(1, p[c, :], 1)[0]
      noise_y[i] = np.where(s == 1)[0]

    actual_noise = (noise_y != y_train).mean()
    assert actual_noise > 0.0

    return noise_y

  n_img = x.shape[0]
  n_classes = 10

  # holdout balanced clean
  val_idx = []
  if n_val > 0:
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  trainlabel = trainlabel.squeeze()
  label_corr_train = trainlabel.copy()

  trainlabel = _generate_asymmetric_noise(trainlabel, noise_ratio)

  if len(trainlabel.shape) == 1:
    trainlabel = np.reshape(trainlabel, [trainlabel.shape[0], 1])

  traindata, trainlabel, label_corr_train = shuffle_dataset(
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


def load_train_val_uniform_noise(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D/2D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  n_img = x.shape[0]
  val_idx = []
  if n_val > 0:
    # Splits a clean holdout set
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  # split validation set
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  # Copies the true label for verification
  label_corr_train = trainlabel.copy()
  # Adds uniform noises
  mask = np.random.rand(len(trainlabel)) <= noise_ratio
  print("Traing index: ",train_idx[:100])
  print("val_idx index: ",val_idx)
  print("Noise index: ",np.where(mask == 1)[0][:100])
  random_labels = np.random.choice(n_classes, mask.sum())
  trainlabel[mask] = random_labels[Ellipsis, np.newaxis]
  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)

def get_clean_index(features,noise_mask):
  all_clean_index = np.where(noise_mask == 0)[0]
  all_noise_index = np.where(noise_mask == 1)[0]
 
  '''
  all_features_gradient = np.matmul(features[1],features[1].T) + np.matmul(features[2],features[2].T)
  print("SHape all features: ",all_features_gradient.shape)
  for i in range(all_features_gradient.shape[0]):
    all_features_gradient[i,i] = 0
  
  all_features_gradient_sum  = np.sum(all_features_gradient,axis = 1)
  '''
  all_features = features[3][0]
  print("SHape all_features 1: ",all_features.shape)
  all_features_2 = features[3][1]
  sorted_value = np.argsort(all_features)
  all_features = np.expand_dims(all_features,axis = 1)
  print("SHape all_features: ",all_features.shape)

  gmm = GaussianMixture(n_components=2,max_iter=30,random_state = 0)
  gmm.fit(all_features)
  pseudo_classification = gmm.predict(all_features) 
  print("ALl mean value: ",gmm.means_)

  clean_set_index = np.argmin(gmm.means_)
  pseudo_classification[pseudo_classification == clean_set_index] = 2
  pseudo_classification[pseudo_classification != 2] = 1
  pseudo_classification[pseudo_classification == 2] = 0
  all_pseudo_clean_index = np.where(pseudo_classification == 0)[0]
  if FLAGS.use_pseudo_label_loss_for_features.startswith("union_GMM_2_loss"):
    all_features_2 = np.expand_dims(all_features_2,axis = 1)
    print("SHape all_features 2: ",all_features_2.shape)

    gmm_2 = GaussianMixture(n_components=2,max_iter=30,random_state = 0,tol=1e-2,reg_covar=5e-4)
    gmm_2.fit(all_features_2)
    pseudo_classification_2 = gmm_2.predict(all_features_2) 
    print("ALl mean value 2: ",gmm_2.means_)

    clean_set_index_2 = np.argmin(gmm_2.means_)
    pseudo_classification_2[pseudo_classification_2 == clean_set_index_2] = 2
    pseudo_classification_2[pseudo_classification_2 != 2] = 1
    pseudo_classification_2[pseudo_classification_2 == 2] = 0
    all_pseudo_clean_index_2 = np.where(pseudo_classification_2 == 0)[0]
    print("Number of all_pseudo_clean_index_2: ",len(all_pseudo_clean_index_2))
    same_element =[ i for i in all_pseudo_clean_index if i in all_pseudo_clean_index_2]
    print("Number of same element: ",len(same_element))
    all_pseudo_clean_index = np.array(list(set(all_pseudo_clean_index) & set(all_pseudo_clean_index_2)))
    print("After Union, number of all_pseudo_clean_index: ",all_pseudo_clean_index.shape)
  
  multiple_fac = 5000
  for i in range(10):
    curr_clean = [j for j in sorted_value[i*multiple_fac:(i+1)*multiple_fac] if j in all_clean_index ]
    curr_noise = [j for j in sorted_value[i*multiple_fac:(i+1)*multiple_fac] if j in all_noise_index ]
    print("First ",(i+1)*multiple_fac," ,Number clean and noise: ",len(curr_clean),"    ",len(curr_noise))
  
  correct_clean_predict = set(all_pseudo_clean_index) & set(all_clean_index)
  print("ACCURACY OF GMM: ",accuracy_score(pseudo_classification,noise_mask))
  print("ACCURACY OF CLEAN: ",len(correct_clean_predict)/len(all_clean_index))
  return all_pseudo_clean_index

def load_train_val_uniform_noise_from_pretrained(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D/2D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  n_img = x.shape[0]
  # Copies the true label for verification
  label_corr_train = y.copy()
  x_noise_mask,y_noise = load_noise_data(x,y.copy(),FLAGS.noise_pretrained)
  
  print("x_noise_mask: ",x_noise_mask[:100])
  print(y_noise[np.where(x_noise_mask == 0)[0]][:100])
  print("y_noise: ",y_noise[:100])
  print(y[np.where(x_noise_mask == 0)[0]][:100])
  print("y: ",y[:100])
  print(y_noise[np.where(x_noise_mask == 0)[0]][:100])
  print(np.squeeze(y[np.where(x_noise_mask == 0)[0]])[:100])
  assert (y_noise[np.where(x_noise_mask == 0)[0]] == np.squeeze(y[np.where(x_noise_mask == 0)[0]])).all()
  val_idx = []
  if n_val > 0:
    # Splits a clean holdout set
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  x_noise_mask = x_noise_mask[train_idx]
  print("First y: ",y)
  if (y != y_noise).any():
    print("They are different")
  
  # split validation set
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
    assert (vallabel == label_corr_train[val_idx]).all()
  traindata, trainlabel, label_corr_train = x[train_idx], y_noise[train_idx], label_corr_train[train_idx]
  
  correc_spot = np.where(x_noise_mask == 0)[0]
  print("Traing index: ",train_idx[:100])
  print("val_idx index: ",val_idx)
  print("Noise index: ",np.where(x_noise_mask == 1)[0][:100])
  print("y train at correct spot: ",trainlabel[correc_spot])
  print("label_corr_train at correct spot: ",label_corr_train[correc_spot])
  
 
  # Shuffles dataset
  traindata, trainlabel, label_corr_train,x_noise_mask = shuffle_dataset(
      traindata, trainlabel, label_corr_train,x_noise_mask)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel),x_noise_mask
def load_noise_data(x,y,path):
  with open(path) as json_file:
    data = json.load(json_file)
  y_noise = y.copy()
  y_noise = np.array(data[0]).astype(np.int32)
  mask = np.zeros(50000)
  mask[data[1]] = 1
  
  return mask,y_noise


def load_train_val_uniform_noise_active(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D/2D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  # Copies the true label for verification
  
  label_corr_train = y.copy()
  x_noise_mask,y_noise = load_noise_data(x,y,FLAGS.noise_pretrained)
  traindata, trainlabel, valdata, vallabel,label_corr_train,noise_mask, traindata_clean,trainlabel_clean,train_clean_index,train_index,left_train_clean_index,all_features = trainval_split_active_learning(x, np.array(y_noise),np.array(label_corr_train),x_noise_mask, n_val)
  tf.logging.info("Unique label of return clean idx: ")
  tf.logging.info(np.unique(vallabel,return_counts = True))
  correc_spot = np.where(noise_mask == 0)[0]
  print("y train at correct spot: ",trainlabel[correc_spot])
  print("label_corr_train at correct spot: ",label_corr_train[correc_spot])
  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=False)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel) , (traindata_clean,trainlabel_clean),(train_clean_index,train_index,left_train_clean_index),all_features,(x[:FLAGS.num_img],np.array(y_noise)[:FLAGS.num_img])

def get_bucket(project: str, bucket_name: str):
  client= storage.Client(project = project)
  bucket = client.get_bucket(bucket_name)
  return bucket
'''
def download_pretrained(data_dir,project,bucket_name):
  
  bucket_prefix = "dung/pretrained_ieg/pretrained"
  
  tf.logging.info("Downloading data from dst folder to " + data_dir +"...")
  

  bucket = get_bucket(project, bucket_name)
  download_folder(bucket,bucket_prefix,data_dir)


  
def download_folder(bucket,bucket_prefix,data_dir):
  dst_folder = Path(data_dir) 
  dst_folder.mkdir(parents = True)
  blobs = list(bucket.list_blobs(prefix = bucket_prefix))

  
  print("ALl blob: ",blobs)
  num_path = len(bucket_prefix.split("/"))
  for blob in tqdm(blobs):
    print("Curr blob: ",blob.name)
    if blob.name.endswith("/"):
      if blob.name != bucket_prefix and blob.name != (bucket_prefix +"/"):
        new_dst_folder = data_dir+ "/"+ blob.name.split("/")[-2]
        print("new_dst_folder: ",new_dst_folder)
        download_folder(bucket,blob.name[:-1],new_dst_folder)
    elif len(blob.name.split("/")) == num_path+1:
      stem = blob.name.split("/")[-1]
      print("Curr stem: ",stem)
      dst_filepath = dst_folder / stem
      with open(dst_filepath,"wb") as sink:
        blob.download_to_file(sink)
'''
def download_pretrained(data_dir,project,bucket_name,bucket_prefix):
  
  
  
  dst_folder = Path(data_dir) 
  if dst_folder.exists():
    tf.logging.info("Skip download data as folder already exist...")
    return
  dst_folder.mkdir(parents = True)
  tf.logging.info(f"Downloading data from dst folder to '{dst_folder}' ...")
  bucket = get_bucket(project, bucket_name)
  blobs = list(bucket.list_blobs(prefix = bucket_prefix))
  
  for blob in tqdm(blobs):
    curr_name = "/".join(blob.name.split("/")[:-1])
    if not blob.name.endswith("/") and curr_name == bucket_prefix:
      stem = blob.name.split("/")[-1]
      
      dst_filepath = dst_folder / stem
      with open(dst_filepath,"wb") as sink:
        blob.download_to_file(sink)
        

def upload_from_directory(directory_path, bucket, dest_blob_name):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    for local_file in rel_paths:

        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[6:])}'
        print()
        print("Uploading from ",local_file," to ",remote_path)
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
def upload_checkpoint(project: str, bucket_name: str,checkpoint_filepath: Union[Path,str]):
  import urllib.parse

  save_checkpoint_filepath = checkpoint_filepath.split("/")[-2]
  print("Current path: ",os.getcwd())
  upload_checkpoint_filepath =   os.getcwd() + "/"
  upload_checkpoint_filepath = upload_checkpoint_filepath + "/".join(checkpoint_filepath.split("/")[:-1])
  bucket_prefix = "dung/ieg_original"
  dst_path = f"{bucket_prefix}/{save_checkpoint_filepath}"

  tf.logging.info(f"Uploading '{upload_checkpoint_filepath}' => '{dst_path}'")
  bucket = get_bucket(project,bucket_name)
  upload_from_directory(str(upload_checkpoint_filepath),bucket,dst_path)
# aquisition function

class CIFAR(object):
  """CIFAR dataset class."""

  def __init__(self):
    self.dataset_name = FLAGS.dataset
    self.is_cifar100 = 'cifar100' in self.dataset_name
    self.num_probe_per_class = 20
    self.new_logits = False
    if self.is_cifar100:
      self.num_classes = 100
    else:
      self.num_classes = 10
    self.noise_ratio = float(self.dataset_name.split('_')[-1])
    assert self.noise_ratio >= 0 and self.noise_ratio <= 1,\
        'The schema {} of dataset is not right'.format(self.dataset_name)
    self.split_probe = FLAGS.probe_dataset_hold_ratio != 0
    self.all_logits = []
    self.clean_index = []
    self.clean_relabel = []
    self.before_after = [] 
    self.dropout_eval_output = []
    self.previous = 1
    print("Number of class is: ",self.num_classes)
    if FLAGS.gcloud:
      download_pretrained("data-ieg-active",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = "dung/pretrained_ieg/pretrained")
      download_pretrained("data-ieg-active/resnet29",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = "dung/pretrained_ieg/pretrained/pretrained_20k/resnet29")
      download_pretrained("data-ieg-active/resnet29/eval",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = "dung/pretrained_ieg/pretrained/pretrained_20k/resnet29/eval")
      download_pretrained("data-ieg-active/resnet29/train",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = "dung/pretrained_ieg/pretrained/pretrained_20k/resnet29/eval")
      
  def get_BALD(self):
    self.BALD_information = []
    pc = self.dropout_eval_output.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(
          axis=-1
    )  # To avoid division with zero, add 1e-10
    E = -np.mean(np.sum(self.dropout_eval_output * np.log(self.dropout_eval_output + 1e-10), axis=-1), axis=0)
    acquisition = H - E
    self.BALD_information = acquisition
  def get_var_ratio(self):
    self.var_ratio_information = []
    #print("self.dropout_eval_output: ",self.dropout_eval_output.shape)
    preds = np.argmax(self.dropout_eval_output, axis=2)
    #print("preds shape: ",preds.shape)
    _, count = stats.mode(preds, axis=0)
    #print("count shape: ",count.shape)
    #print(count)
    acquisition = (1 - count / preds.shape[0]).reshape((-1,))
    #print("acquisition: ",acquisition.shape)
    #print(acquisition[:100])
    self.var_ratio_information = acquisition
  def generator_train(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.train_data.shape[0]
      curr_index += 1
      yield self.train_data[curr_element_index],self.train_label[curr_element_index]
         
  def generator_probe(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.probe_data.shape[0]
      curr_index += 1
      yield self.probe_data[curr_element_index],self.probe_label[curr_element_index]
    
  def generator_reset(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.all_data.shape[0]
      curr_index += 1
      yield self.all_data[curr_element_index],self.all_data_label[curr_element_index]
  def generator_dropout(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.all_data.shape[0]
      curr_index += 1
      yield self.all_data[curr_element_index],self.all_data_label[curr_element_index]
  def get_new_probe(self,features,use_weight = None):
    num_sample_by_class = int(np.floor(FLAGS.num_clean/self.num_classes))
    sim_matrix = cosine_similarity(features,features)
    for i in range(sim_matrix.shape[0]):
      sim_matrix[i,i] = 0
    if use_weight is not None:
      sim_matrix = sim_matrix*use_weight
    
    all_sum_matrix = np.sum(sim_matrix,axis = 1)
    curr_coup = np.argsort(all_sum_matrix)[-num_sample_by_class:]
    return curr_coup

  def get_new_probe_pseudo_label(self,features_clean,features_noisy,class_idx = None,weight_clean = None,weight_noisy = None,curr_class_logits = None,true_label = None):
    num_sample_by_class = int(np.floor(FLAGS.num_clean/self.num_classes))
    if curr_class_logits is not None and "top_clean" in FLAGS.update_probe:
      top_confidence_idx = np.where(curr_class_logits == class_idx)[0]
      remove_label =true_label[np.where(curr_class_logits != class_idx)[0]]
      print("Removing ",(features_clean.shape[0] - len(top_confidence_idx)),"  from this class,only ",len(top_confidence_idx)," left...")
      print("Removing list of class ",class_idx,":",np.unique(remove_label,return_counts = True))
      print()
      features_clean = features_clean[top_confidence_idx]
    sim_matrix = cosine_similarity(features_clean,features_noisy)
    
    if weight_clean is not None:
      sim_matrix = sim_matrix * weight_clean
    if weight_noisy is not None:
      sim_matrix = sim_matrix * weight_noisy
    
     
    all_sum_matrix = np.sum(sim_matrix,axis = 1)
    
    curr_coup = np.argsort(all_sum_matrix)[-num_sample_by_class:]
    if curr_class_logits is not None and "top_clean" in FLAGS.update_probe:
      curr_coup = top_confidence_idx[curr_coup]
    if 'mix' in FLAGS.update_probe:
      internal_sim_matrix = cosine_similarity(features_clean,features_clean)
      internal_sum_matrix = np.sum(internal_sim_matrix,axis = 1)
      curr_coup_internal = np.argsort(internal_sum_matrix)[-num_sample_by_class:]
      final_coup = [curr_coup_internal[-1]]
      curr_choice = -1
      change = 0
      while len(final_coup) < num_sample_by_class:
        
        if curr_coup_internal[curr_choice] not in final_coup:
          final_coup.append(curr_coup_internal[curr_choice])
        else:
          change += 1
        curr_choice -= 1
          
        if change > 100:
          raise ValueError('A very specific bad thing happened.')
      return final_coup
    return curr_coup
  def reset_probe(self,new_label,update_label_idx,new_features = None,all_logits = None,all_loss = None,update_probe = True,lr = 1):
    size_before = len(self.clean_index)
    size_add = len(update_label_idx)

    self.all_data_label[update_label_idx] = new_label
    self.clean_index.extend(update_label_idx)
    size_after = len(self.clean_index)
    assert size_after == size_before + size_add
    self.noise_index = list(set(self.noise_index) - set(update_label_idx))
    
    if "rigged_all" in FLAGS.update_probe:
        tf.logging.info("Using rigged......")
        temp_clean_index = np.where(self.all_data_label == self.all_data_label_correct)[0]
        temp_noise_index = np.where(self.all_data_label != self.all_data_label_correct)[0]
        self.clean_index = list(temp_clean_index)
        self.noise_index = list(temp_noise_index)
        print("Size of clean and noisy after: ",len(self.clean_index),"  ",len(self.noise_index))
    if "rigged_part" in FLAGS.update_probe:
        tf.logging.info("Using rigged part.....")
        temp_clean_index = [i for i in self.clean_index if i in np.where(self.all_data_label == self.all_data_label_correct)[0]]
        temp_noise_index = [i for i in range(self.all_data.shape[0]) if i not in temp_clean_index]
        
        print("Size of clean and noisy after: ",len(temp_clean_index),"  ",len(temp_noise_index))
    
        
        print("Size of clean and noisy after: ",len(temp_clean_index),"  ",len(temp_noise_index))
    
    all_predict_proba = np.max(all_logits,axis = 1) 
    all_predict = np.argmax(all_logits,axis = 1) 
    if len(new_label) > 0:
      curr_acc = accuracy_score(new_label,self.all_data_label_correct[update_label_idx])
      self.all_adding_acc.append([len(update_label_idx),curr_acc])
    else:
      self.all_adding_acc.append([0,0])
    if "new_features" in FLAGS.update_probe:
      tf.logging.info("Changing to using new features")
      self.all_data_features[0] = new_features   
    curr_uncertainty = np.maximum(0,0.5 - self.all_uncertainty_training)
    print("Total uncertainty is: ",curr_uncertainty.shape,"  ",np.mean(curr_uncertainty),"   ",np.median(curr_uncertainty),"   ",np.min(curr_uncertainty),"   ",np.max(curr_uncertainty))
    print("Num uncertainty 0: ",len(np.where(curr_uncertainty == 0)[0]))
    

    if update_probe:
      all_clean_label = self.all_data_label[self.clean_index]
      if "rigged" in FLAGS.update_probe:
        all_clean_label = self.all_data_label_correct[temp_clean_index]
      all_idx_by_class = [np.where(all_clean_label == curr_class)[0] for curr_class in range(self.num_classes)]
      all_probe_idx = []
      new_active_train_index = []
      
      for curr_class in range(self.num_classes):
        if 'weight' in FLAGS.update_probe:
          tf.logging.info("Changing weight probe")
          new_probe_index = self.get_new_probe(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],use_weight = (1 - all_predict_proba[self.clean_index][all_idx_by_class[curr_class]]))
          assert (all_predict_proba >= 0).all() and (all_predict_proba <=1).all()
          print("Finish changing weight probe")
        elif 'random' in FLAGS.update_probe:
          tf.logging.info("Using random probe")
          total_sample_curr_class = len(all_idx_by_class[curr_class])
          num_sample_by_class = int(np.floor(FLAGS.num_clean/self.num_classes))
          new_probe_index = np.random.choice([i for i in range(total_sample_curr_class)], num_sample_by_class, replace=False)
        elif "max_sim_pseudo" in FLAGS.update_probe:
          assert  len(list(set(self.left_noise_index_knn).difference(set(self.noise_index))))  == 0
          all_index_by_class_noisy = [np.where(self.pseudo_knn_label == curr_class)[0] for curr_class in range(self.num_classes)]
          #tf.logging.info("Using max_sim_pseudo")
          if "normal_active_uncertainty" in FLAGS.update_probe:
            #print("Using normal_active_uncertainty")
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
            num_uncertainty_per_class = int(FLAGS.update_probe.split("-")[0].split("_")[-1])
            all_index_by_class_all = [np.where(temp_label == curr_class)[0] for curr_class in range(self.num_classes)]
            curr_class_uncertainty = np.argsort(self.all_uncertainty_training[all_index_by_class_all[curr_class]])
            curr_class_uncertainty = all_index_by_class_all[curr_class][curr_class_uncertainty[:num_uncertainty_per_class]]
            curr_class_logits = np.array(all_predict)[self.clean_index][all_idx_by_class[curr_class]]
            true_label = self.all_data_label_correct[self.clean_index][all_idx_by_class[curr_class]]
            #print("Confirm curr class: ",curr_class)
            #print(np.unique(self.all_data_label_correct[curr_class_uncertainty],return_counts = True))
            if "lr_0_easy" in FLAGS.update_probe: #and lr == 0:
              #print("Using lr_0_easy")
              if "fluctuate" in FLAGS.update_probe:
                if self.previous == 0:
                  curr_class_all_idx = list(set(np.array(self.clean_index)[all_idx_by_class[curr_class]]) | set(curr_class_uncertainty))
                  noisy_features = self.all_data_features[0][curr_class_all_idx]
                  print("Change to all at lr ",lr)
                else:
                  noisy_features = self.all_data_features[0][curr_class_uncertainty]
                  print("Change to only noise at lr ",lr)
              else:
                print("Using normal lr_0_easy")
                curr_class_all_idx = list(set(np.array(self.clean_index)[all_idx_by_class[curr_class]]) | set(curr_class_uncertainty))
                noisy_features = self.all_data_features[0][curr_class_all_idx]
            else:
              noisy_features = self.all_data_features[0][curr_class_uncertainty]
            new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],noisy_features,class_idx = curr_class,curr_class_logits = curr_class_logits,true_label = true_label)
            new_active_train_index.extend(list(curr_class_uncertainty))
          elif "normal_active_BALD" in FLAGS.update_probe:
            print("Using normal_active_BALD")
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
            num_uncertainty_per_class = int(FLAGS.update_probe.split("-")[0].split("_")[-1])
            all_index_by_class_all = [np.where(temp_label == curr_class)[0] for curr_class in range(self.num_classes)]
            self.get_BALD()
            curr_class_uncertainty = np.argsort(self.BALD_information[all_index_by_class_all[curr_class]])
            curr_class_uncertainty = all_index_by_class_all[curr_class][curr_class_uncertainty[-num_uncertainty_per_class:]]

            curr_class_logits = np.array(all_predict)[self.clean_index][all_idx_by_class[curr_class]]
            true_label = self.all_data_label_correct[self.clean_index][all_idx_by_class[curr_class]]
            print("Confirm curr class: ",curr_class)
            print(np.unique(self.all_data_label_correct[curr_class_uncertainty],return_counts = True))
            new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][curr_class_uncertainty],class_idx = curr_class,curr_class_logits = curr_class_logits,true_label = true_label)
            new_active_train_index.extend(list(curr_class_uncertainty))
          elif "normal_active_var_ratio" in FLAGS.update_probe:
            print("Using normal_active_var_ratio")
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
            num_uncertainty_per_class = int(FLAGS.update_probe.split("-")[0].split("_")[-1])
            all_index_by_class_all = [np.where(temp_label == curr_class)[0] for curr_class in range(self.num_classes)]
            self.get_var_ratio()
            #print("self.var_ratio_information: ") 
            #print(self.var_ratio_information[:100])
            curr_class_uncertainty = np.argsort(self.var_ratio_information[all_index_by_class_all[curr_class]])
            curr_class_uncertainty = all_index_by_class_all[curr_class][curr_class_uncertainty[-num_uncertainty_per_class:]]

            curr_class_logits = np.array(all_predict)[self.clean_index][all_idx_by_class[curr_class]]
            true_label = self.all_data_label_correct[self.clean_index][all_idx_by_class[curr_class]]
            print("Confirm curr class: ",curr_class)
            print(np.unique(self.all_data_label_correct[curr_class_uncertainty],return_counts = True))
            new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][curr_class_uncertainty],class_idx = curr_class,curr_class_logits = curr_class_logits,true_label = true_label)
            new_active_train_index.extend(list(curr_class_uncertainty))
          elif "normal_active_only_uncertainty" in FLAGS.update_probe:
            print("Using normal_active_only_uncertainty")
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
            all_index_by_class_all = [np.where(temp_label == curr_class)[0] for curr_class in range(self.num_classes)]
            curr_class_uncertainty = np.argsort(self.all_uncertainty_training[all_index_by_class_all[curr_class]])
            assert (self.all_uncertainty_training >= 0).all() and (self.all_uncertainty_training <= 1).all()
            curr_class_uncertainty = all_index_by_class_all[curr_class][curr_class_uncertainty[:50]]
            print("Confirm curr class: ",curr_class)
            print(np.unique(self.all_data_label_correct[curr_class_uncertainty],return_counts = True))
            new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][curr_class_uncertainty],class_idx = curr_class)
            new_active_train_index.extend(list(curr_class_uncertainty))
          elif "normal_active_mix_uncertainty" in FLAGS.update_probe:
            print("Using normal_active_mix_uncertainty")
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
            all_index_by_class_all = [np.where(temp_label == curr_class)[0] for curr_class in range(self.num_classes)]
            curr_class_uncertainty = np.argsort(self.all_uncertainty_training[all_index_by_class_all[curr_class]])
            assert (self.all_uncertainty_training >= 0).all() and (self.all_uncertainty_training <= 1).all()
            
            
            if "normal_active_mix_uncertainty_cluster" in FLAGS.update_probe:
              num_cluster = 20
              if len(all_idx_by_class[curr_class]) > 2*num_cluster:
                print("Using normal_active_mix_uncertainty_cluster .......")
                if "kmeans" in FLAGS.update_probe:
                  cluster_model = KMeans(n_clusters=num_cluster, random_state=0,init='k-means++').fit(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]])
                  centers = cluster_model.cluster_centers_ 
                  center_indice,_ = vq(centers,self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]])
                elif "kmedoids" in FLAGS.update_probe:
                  cluster_model = KMedoids(n_clusters=num_cluster,metric = "cosine",init='k-medoids++', random_state=0).fit(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]])
                  centers = cluster_model.cluster_centers_ 
                  center_indice,_ = vq(centers,self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]])

          
                print("All center_indice:",np.shape(center_indice))


                
                #print("centers: ",center_indice)
                uncertainty_indice = np.argsort(self.all_uncertainty_training[self.clean_index][all_idx_by_class[curr_class]])
                #print("uncertainty_indice: ",uncertainty_indice)
                uncertainty_indice = [i for i in uncertainty_indice if i not in center_indice]
                #print("uncertainty_indice: ",uncertainty_indice)
                uncertainty_indice = np.array(self.clean_index)[all_idx_by_class[curr_class]][uncertainty_indice[:num_cluster]]
                #print("uncertainty_indice: ",uncertainty_indice)
                center_indice = np.array(self.clean_index)[all_idx_by_class[curr_class]][center_indice]
                curr_class_clean_uncertainty = list(set(center_indice) | set(uncertainty_indice))
                #print("curr_class_clean_uncertainty: ",curr_class_clean_uncertainty)
              else:
                curr_class_clean_uncertainty = list(np.array(self.clean_index)[all_idx_by_class[curr_class]])
           
            assert len(curr_class_clean_uncertainty) == len(all_idx_by_class[curr_class]) or len(curr_class_clean_uncertainty) == 2*num_cluster
            new_active_train_index.extend(list(curr_class_clean_uncertainty))
            
          
          elif "reset_all" in FLAGS.update_probe:
            temp_label = self.all_data_label.copy()
            temp_label[self.left_noise_index_knn] = self.pseudo_knn_label

            all_index_by_class_noisy = [np.where(temp_label[self.temp_noise_index] == curr_class)[0] for curr_class in range(self.num_classes)]
            #print("Unique index pseudo label of class ",curr_class," is ",np.unique(self.all_data_label_correct[self.temp_noise_index][all_index_by_class_noisy[curr_class]],return_counts = True))
          else:
            print("Unique index pseudo label of class ",curr_class," is ",np.unique(self.all_data_label_correct[self.left_noise_index_knn][all_index_by_class_noisy[curr_class]],return_counts = True))

          if "reset_all" in FLAGS.update_probe:
            #print("Using reset_all")
            curr_weight = None
            
            if len(all_index_by_class_noisy[curr_class]) == 0:
              print("No noisy sample for this class")
              if "reset_all_probe" in FLAGS.update_probe:
                print("Using weight for reset_all_probe")
                curr_weight = (1 - all_predict_proba[self.clean_index][all_idx_by_class[curr_class]])
              elif "reset_all_uncertainty" in FLAGS.update_probe:
                #print("Using uncertainty")
                curr_weight = np.maximum(0,0.5 - self.all_uncertainty_training)[self.clean_index][all_idx_by_class[curr_class]]
              new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],class_idx = curr_class,weight_clean = curr_weight)
            else:
              if "reset_all_weight" in FLAGS.update_probe:
                curr_weight = (1 - all_predict_proba[self.temp_noise_index][all_index_by_class_noisy[curr_class]])
              elif "reset_all_uncertainty" in FLAGS.update_probe:
                print("Using uncertainty")
                curr_weight = np.maximum(0,0.5 - self.all_uncertainty_training)[self.temp_noise_index][all_index_by_class_noisy[curr_class]]
              new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][self.temp_noise_index][all_index_by_class_noisy[curr_class]],class_idx = curr_class,weight_noisy = curr_weight)
          else:
            if len(all_index_by_class_noisy[curr_class]) == 0:
              print("No noisy sample for this class")
              new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],class_idx = curr_class)
            else:
              new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],self.all_data_features[0][self.left_noise_index_knn][all_index_by_class_noisy[curr_class]],class_idx = curr_class)
          assert len(new_probe_index) == int(FLAGS.num_clean/self.num_classes)
        else:
          tf.logging.info("Changing normal probe")
          new_probe_index = self.get_new_probe(self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]],use_weight = None)
          print("Finish changing normal probe")
        #print("Adding clean index of current class")
        all_probe_idx.extend(np.array(self.clean_index)[all_idx_by_class[curr_class]][new_probe_index]) 
      if "lr_0_easy" in FLAGS.update_probe and "fluctuate" in FLAGS.update_probe:
        if self.previous == 0:
          self.previous = 1
        else:
          self.previous = 0
      if "normal_active_mix_uncertainty" in FLAGS.update_probe:
        sorted_idx_uncertainty = np.argsort(self.all_uncertainty_training)
        print("new_active_train_index: ",len(new_active_train_index))
        all_uncertainty = [index for index in sorted_idx_uncertainty if index not in new_active_train_index]
        selected_uncertainty = all_uncertainty[:self.num_classes*(2*num_cluster)]
        new_active_train_index.extend(selected_uncertainty)
        all_idx_by_class_left = [[index for index in np.array(self.clean_index)[all_idx_by_class[curr_class]] if index not in new_active_train_index] for curr_class in range(self.num_classes)]
        all_idx_by_class_selected= [np.where(self.all_data_label[new_active_train_index] == curr_class)[0] for curr_class in range(self.num_classes)]
        for curr_class in range(self.num_classes):
            
            new_probe_index = self.get_new_probe_pseudo_label(self.all_data_features[0][all_idx_by_class_left[curr_class]],self.all_data_features[0][np.array(new_active_train_index)[all_idx_by_class_selected[curr_class]]],class_idx = curr_class)
            
            all_probe_idx.extend(np.array(np.array(all_idx_by_class_left)[curr_class])[new_probe_index]) 
      all_train_idx = [idx for idx in range(FLAGS.num_img) if idx not in all_probe_idx]

      if "normal_active_uncertainty" in FLAGS.update_probe:
        print("Using normal_active_uncertainty")
        print("Size before merge: ",len(self.clean_index),"   ",len(new_active_train_index))
        if "reduce_clean" in FLAGS.update_probe:
          reduce_threshold = float(FLAGS.update_probe.split("-")[0].split("_")[-1])
          remove_list = np.where(all_logits >= reduce_threshold)[0]
          print("Removing ",len(remove_list)," from clean set.....")
          all_train_idx = [idx for idx in self.clean_index if idx not in remove_list]
          all_train_idx = list(set(self.clean_index) | set(new_active_train_index))
        else:
          all_train_idx = list(set(self.clean_index) | set(new_active_train_index))
        all_train_idx = [index for index in all_train_idx if index not in all_probe_idx]
        print("Size after merge: ",len(all_train_idx))
      elif "normal_active_only_uncertainty" in FLAGS.update_probe:
        print("Using only_normal_active_uncertainty")
        all_train_idx = list(new_active_train_index)
        all_train_idx = [index for index in all_train_idx if index not in all_probe_idx]
      elif "normal_active_mix_uncertainty" in FLAGS.update_probe:
        print("Using mix_normal_active_uncertainty")
        all_train_idx = list(new_active_train_index)
        all_train_idx = [index for index in all_train_idx if index not in all_probe_idx]
      print("Unique label by class probe: ",np.unique(self.all_data_label[all_probe_idx],return_counts = True))
      self.probe_data,self.probe_label = shuffle_dataset(self.all_data[all_probe_idx],self.all_data_label[all_probe_idx],class_balanced = False)
      
      self.train_data,self.train_label = shuffle_dataset(self.all_data[all_train_idx],self.all_data_label[all_train_idx])
      self.train_index = all_train_idx
      print("Finish update new probe")
    else:
      print("Starting reinitialize  probe")
      self.train_data,self.train_label = shuffle_dataset(self.all_data[self.train_index],self.all_data_label[self.train_index])
      print("Finish reinitialize  probe")
  def get_selected_img(self,num_img_per_class):
    all_selected_idx = []
    all_clean_label = self.all_data_label[self.clean_index]
    all_idx_by_class = [np.where(all_clean_label == curr_class)[0] for curr_class in range(self.num_classes)]
    for curr_class in range(self.num_classes):
      curr_clean_feature = self.all_data_features[0][self.clean_index][all_idx_by_class[curr_class]]
      curr_sim_matrix = cosine_similarity(curr_clean_feature,curr_clean_feature)
      curr_sim_matrix = np.sum(curr_sim_matrix,axis = 1)
      select_index = np.argsort(curr_sim_matrix)[-num_img_per_class:]
      all_selected_idx.append(all_idx_by_class[curr_class][select_index])
    return all_selected_idx

  def create_loader(self):
    
    self.all_adding_acc = []
    self.all_removing_acc = []
    """Creates loader as tf.data.Dataset."""
    # load data to memory.
    if self.is_cifar100:
      (x_train, y_train), (x_test,
                           y_test) = tf.keras.datasets.cifar100.load_data()
    else:
      (x_train, y_train), (x_test,
                           y_test) = tf.keras.datasets.cifar10.load_data()

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_train_original_correct = y_train.copy()[:FLAGS.num_img]
    n_probe = int(math.floor(x_train.shape[0] * FLAGS.probe_dataset_hold_ratio))
    self.output_shapes = ((32, 32, 3),(None))

    # TODO(zizhaoz): add other noise types.
    if 'asymmetric' in self.dataset_name:
      assert 'cifar100' not in self.dataset_name, 'Asymmetric only has CIFAR10'
      if not FLAGS.active:
        (x_train, y_train, y_gold), (x_probe, y_probe) = load_asymmetric(
            x_train,
            y_train,
            noise_ratio=self.noise_ratio,
            n_val=n_probe,
            random_seed=FLAGS.seed)
      else:
        (x_train, y_train, y_gold), (x_probe, y_probe) = load_asymmetric_active(
            x_train,
            y_train,
            noise_ratio=self.noise_ratio,
            n_val=n_probe,
            random_seed=FLAGS.seed)
    elif 'uniform' in self.dataset_name:
      if not FLAGS.active:
        if FLAGS.pretrained_noise:
          tf.logging.info("Loading pretrained noise ............")
          (x_train, y_train, y_gold), (x_probe,
                                       y_probe),noise_mask = load_train_val_uniform_noise_from_pretrained(
                                           x_train,
                                           y_train,
                                           n_classes=self.num_classes,
                                           noise_ratio=self.noise_ratio,
                                           n_val=n_probe)
        else:
          (x_train, y_train, y_gold), (x_probe,
                                       y_probe) = load_train_val_uniform_noise(
                                           x_train,
                                           y_train,
                                           n_classes=self.num_classes,
                                           noise_ratio=self.noise_ratio,
                                           n_val=n_probe)
      else:
        (x_train, y_train, y_gold), (x_probe,
                                     y_probe), (x_train_clean, y_train_clean),(train_clean_index,train_index,left_train_clean_index),all_features,(x_train_original, y_train_original) = load_train_val_uniform_noise_active(
                                         x_train,
                                         y_train,
                                         n_classes=self.num_classes,
                                         noise_ratio=self.noise_ratio,
                                         n_val=n_probe)
    else:
      assert self.dataset_name in ['cifar10', 'cifar100']
    
    
    if not self.split_probe and x_probe is not None:
      # Usually used for supervised comparison.
      tf.logging.info('Merge train and probe')
      x_train = np.concatenate([x_train, x_probe], axis=0)
      y_train = np.concatenate([y_train, y_probe], axis=0)
      y_gold = np.concatenate([y_gold, y_probe], axis=0)

    conf_mat = sklearn_metrics.confusion_matrix(y_gold, y_train)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    tf.logging.info('Corrupted confusion matirx\n {}'.format(conf_mat))
    x_test, y_test = shuffle_dataset(x_test, y_test)
    self.train_dataset_size = x_train.shape[0]
    self.val_dataset_size = x_test.shape[0]
    if self.split_probe:
      self.probe_size = x_probe.shape[0]
    self.train_data,self.train_label = shuffle_dataset(x_train, y_train.squeeze())
    self.probe_data,self.probe_label = shuffle_dataset(x_probe,y_probe.squeeze(),class_balanced = False)
    
    
    if FLAGS.active:
      self.all_data,self.all_data_label,self.all_data_label_correct= x_train_original,y_train_original,np.squeeze(y_train_original_correct)
    #write_to_file(self.train_data,self.train_label,self.probe_data,self.probe_label,self.all_data,self.all_data_label,self.data_path)   
    print("Size of probe data:",self.probe_data.shape)
    print("Size of train data: ",self.train_data.shape)
    
    self.train_dataflow = self.create_ds(self.generator_train,is_train=True)
    self.val_dataflow = self.create_ds((x_test, y_test.squeeze()),
                                       is_train=False,using_generator = False)
    if FLAGS.active:
      input_tuple_clean = tuple(shuffle_dataset(x_train_clean, y_train_clean.squeeze()))
      print("Size of train clean data: ",x_train_clean.shape)
      self.all_data_features  = all_features
      print("Shape all feature:",self.all_data_features[0].shape,self.all_data_features[1].shape,self.all_data_features[2].shape)
      
      self.train_dataflow_clean = self.create_ds(input_tuple_clean, is_train=True,using_generator = False)
    self.clean_index,self.noise_index,self.train_index = list(train_clean_index),[i for i in range(FLAGS.num_img) if i not in train_clean_index],list(left_train_clean_index)
    if FLAGS.update_probe:
      
      
      self.train_data,self.train_label = shuffle_dataset(x_train_clean, y_train_clean.squeeze())
      print("Size of all data and index: ",self.all_data.shape,"  ",np.shape(self.all_data_label),"  ",np.shape(y_train_original))
      print("Index: ",len(self.clean_index)," ",len(self.noise_index)," ",len(self.train_index))  
        

      
      self.train_dataflow_reset = self.create_ds(self.generator_reset,is_train=False)
      self.train_dataflow_dropout = self.create_ds(self.generator_dropout,is_train=False)
    if self.split_probe:
      
      self.probe_dataflow = self.create_ds(self.generator_probe,is_train=True)
    
    selected_img_idx = self.get_selected_img(2)
    self.num_probe_per_class = min([len(curr_class) for curr_class in selected_img_idx])
    selected_img_idx_new = []
    for curr_class in selected_img_idx:
      selected_img_idx_new.extend(curr_class[-self.num_probe_per_class:])
    selected_img_idx_new = np.array(self.clean_index)[selected_img_idx_new]
    print("Total selected image: ",len(selected_img_idx_new))
    self.selected_img,self.selected_img_label = self.all_data[selected_img_idx_new],self.all_data_label[selected_img_idx_new]
    self.selected_img_dataflow = self.create_ds((self.selected_img, self.selected_img_label.squeeze()),
                                       is_train=False,using_generator = False)

    tf.logging.info('Init [{}] dataset loader'.format(self.dataset_name))
    verbose_data('train', x_train, y_train)
    verbose_data('select image', self.selected_img, self.selected_img_label)
    if FLAGS.active:
      verbose_data('train_clean', x_train_clean, y_train_clean)
    verbose_data('test', x_test, y_test)
    if self.split_probe:
      verbose_data('probe', x_probe, y_probe)

    return self

  def create_ds(self, generator, is_train=True,using_generator = True):
    """Creates tf.data object given data.

    Args:
      data: data in format of tuple, e.g. (data, label)
      is_train: bool indicate train stage the original copy, so the resulting
        tensor is 5D

    Returns:
      An tf.data.Dataset object
    """
    if using_generator:
      ds = tf.data.Dataset.from_generator(generator,(tf.int64,tf.int64),output_shapes = self.output_shapes )
    else:
      ds = tf.data.Dataset.from_tensor_slices(generator)

    map_fn = lambda x, y: (cifar_process(x, is_train), y)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


class WebVision(object):
  """Webvision dataset class."""

  def __init__(self, root, version='webvisionmini', use_imagenet_as_eval=False):
    self.version = version
    self.num_classes = 50 if 'mini' in version else 1000
    self.root = root
    self.image_size = 224
    self.use_imagenet_as_eval = use_imagenet_as_eval

    default_n_per_class = 10
    if '_' in FLAGS.dataset:
      self.probe_size = int(FLAGS.dataset.split('_')[1]) * self.num_classes
    else:
      # Uses default ones, assume there is a dataset saved
      self.probe_size = default_n_per_class * self.num_classes
    self.probe_folder = 'probe_' + str(self.probe_size)

  def wrapper_map_probe_v2(self, tfrecord):
    """tf.data.Dataset map function for probe data v2.

    Args:
      tfrecord: serilized by tf.data.Dataset.

    Returns:
      A map function
    """

    def _extract_fn(tfrecord):
      """Extracts the functions."""

      features = {
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/label': tf.FixedLenFeature([], tf.int64)
      }
      example = tf.parse_single_example(tfrecord, features)
      image, label = example['image/encoded'], tf.cast(
          example['image/label'], dtype=tf.int32)

      return [image, label]

    image_bytes, label = _extract_fn(tfrecord)
    label = tf.cast(label, tf.int64)

    image = imagenet_preprocess_image(
        image_bytes, is_training=True, image_size=self.image_size)

    return image, label

  def wrapper_map_v2(self, train):
    """tf.data.Dataset map function for train data v2."""

    def _func(data):
      img, label = data['image'], data['label']
      image_bytes = tf.image.encode_jpeg(img)
      image_1 = imagenet_preprocess_image(
          image_bytes, is_training=train, image_size=self.image_size)
      if train:
        image_2 = imagenet_preprocess_image(
            image_bytes,
            is_training=train,
            image_size=self.image_size,
            autoaugment_name='v0',
            use_cutout=True)
        images = tf.concat(
            [tf.expand_dims(image_1, 0),
             tf.expand_dims(image_2, 0)], axis=0)
      else:
        images = image_1
      return images, label

    return _func

  def create_loader(self):
    """Creates loader."""

    if self.use_imagenet_as_eval:
      # To evaluate on webvision eval, set this to False.
      split = ['train']
      val_ds, imagenet_info = tfds.load(
          name='imagenet2012',
          download=True,
          split='validation',
          data_dir=self.root,
          with_info=True)
      val_info = imagenet_info.splits['validation']
      tf.logging.info('WebVision: use imagenet validation')
    else:
      split = ['train', 'val']
    assert tfds.__version__.startswith('2.'),\
        'tensorflow_dataset version must be 2.x.x to use image_label_folder.'
    ds, self.info = tfds.load(
        'image_label_folder',
        split=split,
        data_dir=self.root,
        builder_kwargs=dict(dataset_name=self.version),
        with_info=True)

    train_info = self.info.splits['train']

    if len(split) == 2:
      train_ds, val_ds = ds
      val_info = self.info.splits['val']
    else:
      train_ds = ds[0]

    self.train_dataset_size = train_info.num_examples
    self.val_dataset_size = val_info.num_examples
    self.test_dataset_size = self.val_dataset_size

    train_ds = train_ds.map(
        self.wrapper_map_v2(True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        self.wrapper_map_v2(False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    self.train_dataflow = train_ds
    self.val_dataflow = val_ds

    def _get_probe():
      """Create probe data tf.data.Dataset."""
      probe_ds = tf.data.TFRecordDataset(
          os.path.join(self.root, self.version, self.probe_folder,
                       'imagenet2012-probe.tfrecord-1-of-1'))
      probe_ds = probe_ds.map(
          self.wrapper_map_probe_v2,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # For single file, we need to disable auto_shard_policy for multi-workers,
      # e.g. every worker takes the same file
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = (
          tf.data.experimental.AutoShardPolicy.OFF)
      probe_ds = probe_ds.with_options(options)

      return probe_ds

    self.probe_dataflow = _get_probe()

    tf.logging.info(self.info)
    tf.logging.info('[{}] Create {} \n train {} probe {} val {}'.format(
        self.version, FLAGS.dataset, self.train_dataset_size,
        self.probe_size, self.val_dataset_size))
    return self

def get_maximize_info_2_improve(sim_matrix,ratio):
    curr_max_count = -999999
    curr_max_sum = -9999
    curr_coup = [-1,-1]
    for i in range(sim_matrix.shape[0]):
        for j in range(i+1,sim_matrix.shape[0]):
            left_idx = [k for k in range(sim_matrix.shape[0]) if k != i and k != j]
            external_value = (sim_matrix[i][left_idx] + sim_matrix[j][left_idx])/2
            curr_count = len(np.where(external_value >= ratio)[0])
            curr_sum = np.sum(external_value)
            if curr_count > curr_max_count or (curr_count == curr_max_count and curr_sum > curr_max_sum):
                curr_max_count = curr_count
                curr_max_sum = curr_sum
                curr_coup = [i,j]
    return curr_coup
def select_representation_test_active(features,num_class = 10,num_point_per_class = 10,correct_label = None,prediction = [],prediction_label = [],prediction_proba = [],noise_mask = [],noisy_label = []):
    FLAGS = tf.flags.FLAGS

  

    all_features = features[0]
    all_features_1 = features[1]
    all_features_2 = features[2]
    correct_label = correct_label[:FLAGS.num_img]
    curr_noise_mask = noise_mask[:FLAGS.num_img]
    
    if FLAGS.use_GMM_pseudo_classification:
      all_clean_index =  get_clean_index(features,curr_noise_mask) 
    else:
      all_clean_index = np.where(curr_noise_mask == 0)[0]
    all_noise_index =  np.array([i for i in range(all_features.shape[0]) if i not in all_clean_index])                  #np.where(curr_noise_mask == 1)[0]
    

  
    
    return all_clean_index[:100],all_clean_index













                       



