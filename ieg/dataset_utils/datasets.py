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
from tensorflow import keras 
from tensorflow.keras import Model,layers
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
import torch.nn.functional as functional

FLAGS = flags.FLAGS


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
def shuffle_dataset(data, label, others=None,noise_mask = None, class_balanced=False,return_index = False):
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
    if return_index:
      return data[ids], label[ids],ids
    else:
      return data[ids], label[ids]
  elif noise_mask is None:
    if return_index:
      return data[ids], label[ids], others[ids],ids
    else:
      return data[ids], label[ids], others[ids]
  else:
    if return_index:
      return data[ids], label[ids], others[ids],curr_noise_mask,ids
    else:
      return data[ids], label[ids], others[ids],curr_noise_mask


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
def load_noise_data(x,y,path,type_noise):
  data_noise = None
  if type_noise == "open":
    noise_index = np.load(path[2])
    data_noise = np.load(path[0])
    y_noise = np.load(path[1])
    mask = np.zeros(50000)
    mask[noise_index] = 1
  elif type_noise == "semantic_pretrained":
    y_noise = np.load(path[1])
    y_noise = np.argmax(y_noise,axis = 1)
    mask = np.zeros(50000)
    mask[np.where(y != y_noise)[0]] = 1
  else:
    with open(path) as json_file:
      data = json.load(json_file)
    y_temp = np.squeeze(y.copy())
    y_noise = np.array(data[0]).astype(np.int32)
    mask = np.zeros(50000)
    noise_mask = np.where( y_noise != y_temp)[0]
    mask[noise_mask] = 1
  
  return mask,y_noise,data_noise


def load_train_val_pretrained_noise_active(x, y, n_classes, n_val, noise_ratio,type_noise):
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
  if type_noise == "openset" or type_noise == "semantic_pretrained":
    if type_noise == "openset":
      path_data = os.path.join(FLAGS.noise_pretrained,"openset_in_c10_out_"+type_openset+".npy") 
      path_label = os.path.join(FLAGS.noise_pretrained,"openset_in_c10_train_labels.npy")
      path_index = os.path.join(FLAGS.noise_pretrained,"noise_index.npy")
      x_noise_mask,y_noise,data_noise = load_noise_data(x,y.copy(),[path_data,path_label,path_index],type_noise = type_noise)
      x = data_noise
    else:
      path_data = os.path.join(FLAGS.noise_pretrained,data+"_train_data.npy")
      path_label = os.path.join(FLAGS.noise_pretrained,data+"_train_label.npy")
      x_noise_mask,y_noise,data_noise = load_noise_data(x,y.copy(),[path_data,path_label],type_noise = type_noise)
  else:
    x_noise_mask,y_noise,_ = load_noise_data(x,y.copy(),FLAGS.noise_pretrained,type_noise = type_noise)
    
    print("Accuracy of noise: ",accuracy_score(y_noise,y))
  print("Number validation: ",n_val)
  traindata, trainlabel, valdata, vallabel,label_corr_train,noise_mask, traindata_clean,trainlabel_clean,train_clean_index,train_index,left_train_clean_index,probe_index,all_features,prediction_label,all_loss = trainval_split_active_learning(x, np.array(y_noise),np.array(label_corr_train),x_noise_mask, n_val)
  tf.logging.info("Unique label of return clean idx: ")
  tf.logging.info(np.unique(vallabel,return_counts = True))
  correc_spot = np.where(noise_mask == 0)[0]
  print("y train at correct spot: ",trainlabel[correc_spot])
  print("label_corr_train at correct spot: ",label_corr_train[correc_spot])
  
 
  # Shuffles dataset
  traindata, trainlabel, label_corr_train,idx = shuffle_dataset(
      traindata, trainlabel, label_corr_train,return_index = True)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel) , (traindata_clean,trainlabel_clean),(train_clean_index,np.array(train_index)[idx],left_train_clean_index,probe_index),all_features,(x[:FLAGS.num_img],np.array(y_noise)[:FLAGS.num_img]),prediction_label,all_loss,x

def get_bucket(project: str, bucket_name: str):
  client= storage.Client(project = project)
  bucket = client.get_bucket(bucket_name)
  return bucket
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
  bucket_prefix = "dung/ieg_original_backup_scaled_simple_new"
  dst_path = f"{bucket_prefix}/{save_checkpoint_filepath}"

  tf.logging.info(f"Uploading '{upload_checkpoint_filepath}' => '{dst_path}'")
  bucket = get_bucket(project,bucket_name)
  upload_from_directory(str(upload_checkpoint_filepath),bucket,dst_path)

class CIFAR(object):
  """CIFAR dataset class."""

  def __init__(self):
    self.dataset_name = FLAGS.dataset
    self.is_cifar100 = 'cifar100' in self.dataset_name
    if self.is_cifar100:
      self.num_classes = 100
    else:
      self.num_classes = 10
    self.noise_ratio = float(self.dataset_name.split('_')[-1])
    assert self.noise_ratio >= 0 and self.noise_ratio <= 1,\
        'The schema {} of dataset is not right'.format(self.dataset_name)
    self.reset_now = True
    self.split_probe = FLAGS.probe_dataset_hold_ratio != 0
    self.probe_index = []
    self.all_probe_acc = []
    self.all_adding_acc = []
    if FLAGS.gcloud:
      print("Downloading data .....")
      download_pretrained("data-ieg-active",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = "dung/pretrained_ieg/pretrained")
      download_pretrained("data-ieg-active/"+str(FLAGS.network_name),FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = FLAGS.pretrained_path+str(FLAGS.network_name))
      download_pretrained("data-ieg-active/"+str(FLAGS.network_name)+"/eval",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = FLAGS.pretrained_path +str(FLAGS.network_name)+"/eval")
      download_pretrained("data-ieg-active/"+str(FLAGS.network_name)+"/train",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = FLAGS.pretrained_path +str(FLAGS.network_name)+"/train")

      if "openset" in self.dataset_name:
        download_pretrained("data-ieg-active/openset_noise",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = FLAGS.pretrained_path +"openset_noise")
      if "semantic" in self.dataset_name:
        download_pretrained("data-ieg-active/data_semantic_noisy",FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,bucket_prefix = FLAGS.pretrained_path +"data_semantic_noisy")
  def update_loss_ema_simple(self,r1,r2,similarity_list):
    tf.logging.info("Parameter receive: "+str(r1)+" "+str(r2))
    onehot_label =  functional.one_hot(torch.Tensor(self.all_data_label).to(torch.int64),num_classes = self.num_classes).numpy()

    def get_ce_loss(x,all_logit,all_onehot,num_logit):
      curr_onehot = np.tile(all_onehot[x],(num_logit[x][0],1))
      ce_loss = curr_onehot*(-torch.log(torch.Tensor(all_logit[x][0][-num_logit[x][0]:]))).numpy()
      mean_loss = np.mean(np.sum(ce_loss,axis = 1))
      return mean_loss
    all_index = np.array([i for i in range(len(self.all_data_label))])
    all_index = np.expand_dims(all_index,axis = 1).astype(int)
    representative_loss = np.apply_along_axis(get_ce_loss, 1,arr = all_index,all_logit = self.all_predict_logit[0],all_onehot = onehot_label,num_logit = self.all_predict_logit[1].astype(int))
    
    assert (representative_loss >=0).all()
    representative_loss  = np.abs(representative_loss)
    print('max,min and mean,std of representative_loss: ',np.max(representative_loss),"  ",np.min(representative_loss),"  ",np.mean(representative_loss),"   ",np.std(representative_loss))
    print('max,min and mean,std of similarity_list: ',np.max(similarity_list),"  ",np.min(similarity_list),"  ",np.mean(similarity_list),"   ",np.std(similarity_list))
    representative_loss_for_min = np.exp(r1*representative_loss)
    similarity_list = np.exp(r2*similarity_list)
    
    representative_loss_for_min_proba = representative_loss_for_min/np.sum(representative_loss_for_min)
    similarity_proba = similarity_list/np.sum(similarity_list)
    if "only_loss" in FLAGS.update_probe:
      self.all_CL_select_proba = representative_loss_for_min_proba
    if "only_sim" in FLAGS.update_probe:
      self.all_CL_select_proba = similarity_proba
    
    self.all_CL_select_proba = np.maximum.reduce([representative_loss_for_min_proba , similarity_proba])  

  def reset_probe(self,new_label,update_label_idx,new_features = None,all_logits = None,all_loss = None,update_probe = True,lr = 1,all_original_logits= None,curtainty_label_matrix = None):

    self.all_data_label[update_label_idx] = new_label
    self.clean_index = list(set(self.clean_index) | set(update_label_idx))
    self.real_acc_train = accuracy_score(self.all_data_label,self.all_data_label_correct)
    self.acc_clean = accuracy_score(self.all_data_label[self.clean_index],self.all_data_label_correct[self.clean_index])
    self.noise_index = list(set(self.noise_index) - set(update_label_idx))
    
   
    if curtainty_label_matrix is not None:
      print("Shape data: ",np.shape(self.all_data_label),np.shape(self.train_label),np.shape(curtainty_label_matrix))
      qualified_sample_idx_2 = np.where(curtainty_label_matrix == True)[0]
      print("Number of qualified_sample_idx_2: ",len(qualified_sample_idx_2))
      true_clean = np.where(self.all_data_label == self.all_data_label_correct)[0]
      final = list(set(qualified_sample_idx_2) & set(true_clean))
      print("Number of true qualified_sample_idx_2: ",len(qualified_sample_idx_2))
    all_predict_proba = np.max(all_logits,axis = 1) 
    all_predict = np.argmax(all_logits,axis = 1) 
    if len(new_label) > 0:
      curr_acc = accuracy_score(new_label,self.all_data_label_correct[update_label_idx])
      self.all_adding_acc.append([len(update_label_idx),curr_acc])
      print("accuracy_score curr_acc: ",curr_acc)
      print("Adding ",len(new_label)," samples")
    else:
      print("Adding 0 samples")
      self.all_adding_acc.append([0,0])

    curr_uncertainty = np.maximum(0,0.5 - self.all_uncertainty_training)
   
    if update_probe:
      all_clean_label = self.all_data_label[self.clean_index]
      all_idx_by_class = [np.where(all_clean_label == curr_class)[0] for curr_class in range(self.num_classes)]
      all_probe_idx = []
      temp_label = self.all_data_label.copy()
      temp_label[self.left_noise_index_knn] = self.pseudo_knn_label
      qualified_sample_idx_2 = np.where(curtainty_label_matrix == True)[0]
      for curr_class in range(self.num_classes):
        new_probe_index = []
        if "max_sim_pseudo" in FLAGS.update_probe:
            curr_all_features = self.all_data_features[0]
          
            if "rigMagMin" in FLAGS.update_probe:
                qualified_sample = list(set(np.where(self.all_data_label_correct[self.clean_index] == curr_class)[0]) & set(np.where(self.all_data_label[self.clean_index] == curr_class)[0]))
                qualified_sample = np.array(self.clean_index)[qualified_sample]
                qualified_sample = np.array(list(set(qualified_sample).difference(self.probe_index)))
                curr_all_mag = all_logits #self.all_predict_logit[0][:,-1,:]
                assert curr_all_mag.shape == (FLAGS.num_img,self.num_classes)
                curr_all_mag = np.sum(curr_all_mag*curr_all_mag,axis = 1)
                qualified_sample_uncer_sort = np.argsort(curr_all_mag[qualified_sample])[:self.num_sample_by_class]
                new_probe_index = qualified_sample[qualified_sample_uncer_sort]
                new_probe_index = list(new_probe_index)
            elif "rigCircleMagMin" in FLAGS.update_probe:
                qualified_sample = list(set(np.where(self.all_data_label_correct[self.clean_index] == curr_class)[0]) & set(np.where(self.all_data_label[self.clean_index] == curr_class)[0]))
                qualified_sample = np.array(self.clean_index)[qualified_sample]
                qualified_sample = list(set(qualified_sample).difference(self.probe_index))
                curr_all_mag = all_logits #self.all_predict_logit[0][:,-1,:]
                assert curr_all_mag.shape == (FLAGS.num_img,self.num_classes)

                curr_all_mag = np.sum(curr_all_mag*curr_all_mag,axis = 1)
                qualified_sample_uncer_sort = np.argsort(curr_all_mag[qualified_sample])[0]
                new_probe_index.append(qualified_sample[qualified_sample_uncer_sort])
            elif "CircleMagMin" in FLAGS.update_probe:
                qualified_sample = np.where(self.all_data_label[self.clean_index] == curr_class)[0]
                qualified_sample = np.array(self.clean_index)[qualified_sample]
                qualified_sample = np.array(list(set(qualified_sample).difference(self.probe_index)))
                qualified_sample_idx = np.where(np.argmax(all_logits[qualified_sample],axis = 1) == curr_class)[0]
                qualified_sample_idx = qualified_sample[qualified_sample_idx]

                qualified_sample_idx = np.array(list(set(qualified_sample_idx) & set(qualified_sample_idx_2))).astype(int)
                qualified_sample = list(set(qualified_sample).difference(self.probe_index))
                curr_all_mag = all_logits # self.all_predict_logit[0][:,-1,:]
                
                assert curr_all_mag.shape == (FLAGS.num_img,self.num_classes)
                curr_all_mag = np.sum(curr_all_mag*curr_all_mag,axis = 1)
                if len(qualified_sample_idx) > 0:
                  
                  qualified_sample_uncer_sort = np.argsort(curr_all_mag[qualified_sample])[0]
                  new_probe_index.append(qualified_sample[qualified_sample_uncer_sort])
                else:
                  tf.logging.info("Class "+ str(curr_class)+" only "+ str(len(new_probe_index))+ str(new_probe_index))
                  curr_class_clean_idx = np.where(self.all_data_label[self.clean_index] == curr_class)[0] 
                  clean_idx_uncertainty = np.argsort(curr_all_mag[np.array(self.clean_index)[curr_class_clean_idx]])
                  clean_idx_uncertainty = np.array(self.clean_index)[curr_class_clean_idx][clean_idx_uncertainty]
                  curr_select_idx = 0
                  while len(new_probe_index) < 1:
                    if clean_idx_uncertainty[curr_select_idx] not in new_probe_index and clean_idx_uncertainty[curr_select_idx] not in self.probe_index:
                      new_probe_index.append(clean_idx_uncertainty[curr_select_idx])
                    curr_select_idx += 1
            elif "MagMin" in FLAGS.update_probe:
                
                qualified_sample = np.where(self.all_data_label[self.clean_index] == curr_class)[0]
                qualified_sample = np.array(self.clean_index)[qualified_sample]

                qualified_sample_idx = np.where(np.argmax(all_logits[qualified_sample],axis = 1) == curr_class)[0]
                qualified_sample_idx = qualified_sample[qualified_sample_idx]
                qualified_sample_idx = np.array(list(set(qualified_sample_idx) & set(qualified_sample_idx_2))).astype(int)
                curr_all_mag = all_logits #self.all_predict_logit[0][:,-1,:]
                assert curr_all_mag.shape == (FLAGS.num_img,self.num_classes)
                curr_all_mag = np.sum(curr_all_mag*curr_all_mag,axis = 1)
                assert len(curr_all_mag) == len(self.all_data_label)
                qualified_sample_uncer_sort = np.argsort(curr_all_mag[qualified_sample_idx])[:self.num_sample_by_class]
                new_probe_index = qualified_sample_idx[qualified_sample_uncer_sort]
                new_probe_index = list(new_probe_index)
                if len(new_probe_index) < self.num_sample_by_class:
                  tf.logging.info("Class "+ str(curr_class)+" only "+ str(len(new_probe_index)))
                  curr_class_clean_idx = np.where(self.all_data_label[self.clean_index] == curr_class)[0] 
                  clean_idx_uncertainty = np.argsort(curr_all_mag[np.array(self.clean_index)[curr_class_clean_idx]])
                  clean_idx_uncertainty = np.array(self.clean_index)[curr_class_clean_idx][clean_idx_uncertainty]
                  curr_select_idx = 0
                  
                  while len(new_probe_index) < self.num_sample_by_class:
                    if clean_idx_uncertainty[curr_select_idx] not in new_probe_index:
                      new_probe_index.append(clean_idx_uncertainty[curr_select_idx])
                    curr_select_idx += 1
        all_probe_idx.extend(new_probe_index) 

      
     

      if "CircleMagMin" in FLAGS.update_probe:
        self.probe_index = np.array(list(self.probe_index[self.num_classes:]) + list(all_probe_idx))
        self.probe_data,self.probe_label = self.all_data[self.probe_index],self.all_data_label[self.probe_index]
        all_probe_idx = self.probe_index.copy()
      else:
         self.probe_data,self.probe_label,self.probe_index = shuffle_dataset(self.all_data[all_probe_idx],self.all_data_label[all_probe_idx],class_balanced = True,return_index = True)
         self.probe_index = np.array(all_probe_idx)[self.probe_index]
      old_probe_index = self.probe_index.copy()
      all_train_idx = [idx for idx in range(FLAGS.num_img) if idx not in self.probe_index]
      self.train_data,self.train_label = self.all_data[all_train_idx],self.all_data_label[all_train_idx]
      self.train_index = np.array(all_train_idx)
      tf.logging.info("Adding probe accuracy....")
      self.all_probe_acc.append(accuracy_score(self.all_data_label[self.probe_index],self.all_data_label_correct[self.probe_index]))

      similarity_list = np.zeros([self.all_data.shape[0]])
      print("Accuracy temp label: ",accuracy_score(self.all_data_label_correct,temp_label))

      for curr_class in range(self.num_classes):
            curr_class_samples = np.where(temp_label == curr_class)[0]
            print("Accuracy class ",curr_class,": ",accuracy_score(self.all_data_label_correct[curr_class_samples],temp_label[curr_class_samples]))
            if "CircleMagMin" in FLAGS.update_probe:
              curr_class_index = [(curr_class + self.num_classes*i) for i in range(self.num_sample_by_class)]
              curr_class_index = np.array(self.probe_index)[curr_class_index]
              assert (self.all_data_label[curr_class_index] == curr_class).all()
              assert len(curr_class_index) == self.num_sample_by_class
              if "last" in FLAGS.using_new_features :
                curr_class_sim = np.max(cosine_similarity(self.milestone_fea[curr_class_samples],self.milestone_fea[curr_class_index]),axis = 1)
              else:
                curr_class_sim = np.max(cosine_similarity(curr_all_features[curr_class_samples],curr_all_features[curr_class_index]),axis = 1)

            else:
              curr_class_index = np.array(all_probe_idx)[(curr_class*self.num_sample_by_class):((curr_class+1)*self.num_sample_by_class)]
              assert (self.all_data_label[curr_class_index] == curr_class).all()
              assert len(curr_class_index) == self.num_sample_by_class
              if "last" in FLAGS.using_new_features :
                cur_class_feature = cosine_similarity(self.milestone_fea[curr_class_samples],self.milestone_fea[np.array(all_probe_idx)[(curr_class*self.num_sample_by_class):((curr_class+1)*self.num_sample_by_class)]])
                curr_class_sim = np.sum(cur_class_feature,axis = 1)
              else:
                curr_class_sim = np.sum(cosine_similarity(curr_all_features[curr_class_samples],curr_all_features[curr_class_index]),axis = 1)
           
            similarity_list[curr_class_samples] = curr_class_sim        
      parameter_sim = 1
      parameter_divergence = 1
      parameter_loss = 1
        
      if "_" in  FLAGS.update_loss:
        parameter_sim = int(FLAGS.update_loss.split("_")[0])
        parameter_divergence = int(FLAGS.update_loss.split("_")[1])
        parameter_loss = int(FLAGS.update_loss.split("_")[2])
      self.update_loss_ema_simple(r1 = -parameter_loss,r2 = parameter_sim,similarity_list = similarity_list)

      self.CL_select_proba = self.all_CL_select_proba[all_train_idx]
      self.CL_select_proba = (self.CL_select_proba.astype(np.float64)/np.sum(self.CL_select_proba).astype(np.float64)).astype(np.float64)

      print("Finish update new probe")
      tf.logging.info("Shape of new probe and train data: "+str(self.train_data.shape)+"   "+str(self.probe_data.shape))
      assert (self.probe_label == self.all_data_label[old_probe_index]).all()
      assert (self.train_label == self.all_data_label[self.train_index]).all()
    self.reset_now = True

  def generator_warmup_probability(self):
    curr_index = 0
    update_probe_interval = 2500000
    all_next_index = np.random.choice([i for i in range(self.train_data.shape[0])],size = update_probe_interval,p = self.CL_select_proba/self.CL_select_proba.sum())
    while True: 
      curr_element_index = all_next_index[curr_index]
      curr_index += 1
      yield self.train_data[curr_element_index],self.train_label[curr_element_index],self.train_index[curr_element_index]
  def generator_train_probability(self):
    self.curr_index = 0
    update_probe_interval = 2500000
    num_time_generate  = 0 
    while True:

      if self.reset_now:
        num_time_generate += 1
        self.all_next_index = np.random.choice([i for i in range(self.train_data.shape[0])],size = update_probe_interval,p = self.CL_select_proba/self.CL_select_proba.sum())
        frequency = np.unique(self.all_next_index,return_counts = True)
        print("num,max,min,standard deviation of number training time: ",len(frequency[0]),np.max(frequency[1]),np.min(frequency[1]),np.std(frequency[1]))
        self.curr_index = 0
        self.reset_now = False
        gt_shape_0 = FLAGS.num_img - self.num_sample_by_class*self.num_classes
        print("self.curr_index: ",self.curr_index)
        assert self.train_data.shape[0] == gt_shape_0 
        assert len(self.train_label) == gt_shape_0 
        assert self.train_index.shape[0] == gt_shape_0 
      curr_element_index = self.all_next_index[self.curr_index]
      self.curr_index += 1
      yield self.train_data[curr_element_index],self.train_label[curr_element_index],self.train_index[curr_element_index]
         
  def generator_probe(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.probe_data.shape[0]
      curr_index += 1

      yield self.probe_data[curr_element_index],self.probe_label[curr_element_index],self.probe_index[curr_element_index]
    
  def generator_reset(self):
    curr_index = 0
    while True:
      curr_element_index = curr_index % self.all_data.shape[0]
      curr_index += 1
      yield self.all_data[curr_element_index],self.all_data_label[curr_element_index]

  def create_loader(self):
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
    n_probe = int(round(FLAGS.num_img * FLAGS.probe_dataset_hold_ratio))
    self.output_clean_shapes = ((32, 32, 3),(None),(None))
    self.output_shapes = ((32, 32, 3),(None))

    # TODO(zizhaoz): add other noise types.
    if 'asymmetric' in self.dataset_name:
      assert 'cifar100' not in self.dataset_name, 'Asymmetric only has CIFAR10'
      if not FLAGS.active:
        if FLAGS.pretrained_noise:
          tf.logging.info("Loading pretrained noise ............")
          (x_train, y_train, y_gold), (x_probe, y_probe),noise_mask = load_train_val_uniform_noise_from_pretrained(
                                             x_train,
                                             y_train,
                                             n_classes=self.num_classes,
                                             noise_ratio=self.noise_ratio,
                                             n_val=n_probe)
        else:
          (x_train, y_train, y_gold), (x_probe, y_probe) = load_asymmetric(x_train,y_train,noise_ratio=self.noise_ratio,n_val=n_probe,random_seed=FLAGS.seed)
      else:
        (x_train, y_train, y_gold), (x_probe,
                                     y_probe), (x_train_clean, y_train_clean),(train_clean_index,train_index,left_train_clean_index,probe_index),all_features,(x_train_original, y_train_original),prediction_label,all_loss,_  = load_train_val_pretrained_noise_active(
                                         x_train,
                                         y_train,
                                         n_classes=self.num_classes,
                                         noise_ratio=self.noise_ratio,
                                         n_val=n_probe,type_noise = "asymmetric")
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
                                     y_probe), (x_train_clean, y_train_clean),(train_clean_index,train_index,left_train_clean_index,probe_index),all_features,(x_train_original, y_train_original),prediction_label,all_loss,_  = load_train_val_pretrained_noise_active(
                                         x_train,
                                         y_train,
                                         n_classes=self.num_classes,
                                         noise_ratio=self.noise_ratio,
                                         n_val=n_probe,type_noise = "uniform")
    else:
      assert self.dataset_name in ['cifar10', 'cifar100']
    
    tf.logging.info("All label of probe:")
    tf.logging.info(",".join([str(label) for label in y_probe]))
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
    self.num_sample_by_class = int(n_probe/self.num_classes)
    if not FLAGS.active:
      input_tuple = (x_train, y_train.squeeze())
      self.train_dataflow = self.create_ds(input_tuple, is_train=True, using_generator = False)
    
    else:
      self.all_data,self.all_data_label,self.all_data_label_correct= x_train_original,y_train_original,np.squeeze(y_train_original_correct)
      self.CL_select_proba = np.array([1/x_train.shape[0] for i in range(x_train.shape[0])])
      

      self.all_data_features  = all_features
      self.prediction_magnitude = np.zeros((self.all_data.shape[0],5))
      
      temp_probe_index = np.reshape(np.array(probe_index),(self.num_classes,self.num_sample_by_class))
      for i in range(self.num_classes):
        assert (self.all_data_label[temp_probe_index[i]] == i).all()
      for i in range(self.num_sample_by_class):
        self.probe_index.extend(temp_probe_index[:,self.num_sample_by_class - i - 1])
      self.clean_index,self.noise_index = list(set(train_clean_index) | set(self.probe_index)),[i for i in range(FLAGS.num_img) if i not in train_clean_index]
      self.real_acc_train = accuracy_score(self.all_data_label,self.all_data_label_correct)
      self.acc_clean = accuracy_score(self.all_data_label[self.clean_index],self.all_data_label_correct[self.clean_index])
      self.all_FE_label = np.array(prediction_label)

      self.probe_data,self.probe_label = self.all_data[self.probe_index],self.all_data_label[self.probe_index]
      self.train_data,self.train_label = self.all_data[train_index], self.all_data_label[train_index]
      self.train_index = np.array(train_index)
      if "soft" not in FLAGS.extra_name:
        image_clean,label_clean,new_idx = shuffle_dataset(x_train_clean, y_train_clean.squeeze(),return_index = True)
        input_tuple_clean =(image_clean,label_clean,np.array(left_train_clean_index)[new_idx])
        self.train_dataflow_clean = self.create_ds_hard_clean(input_tuple_clean, is_train=True,using_generator = False)
      else:
        self.train_dataflow_clean = self.create_ds_hard_clean(self.generator_warmup_probability, is_train=True)
      self.train_dataflow = self.create_ds_hard_clean(self.generator_train_probability,is_train=True)
      self.train_dataflow_reset = self.create_ds(self.generator_reset,is_train=False)
      self.all_predict_logit = [np.zeros((self.all_data.shape[0],5,self.num_classes)),np.zeros(self.all_data.shape[0])]

      if "soft" in FLAGS.extra_name:
        similarity_list = np.zeros(self.all_data.shape[0])
        all_index_by_class = [np.where(self.all_FE_label == curr_class)[0]  for curr_class in range(self.num_classes)]
        parameter_sim = 1
        parameter_loss = 1
        
        if "_" in  FLAGS.start_proba:
          parameter_sim = int(FLAGS.start_proba.split("_")[0])
          parameter_loss = int(FLAGS.start_proba.split("_")[1])

        for curr_class in range(self.num_classes):
          assert (self.all_data_label[temp_probe_index[curr_class]] == curr_class).all()
          sum_sim = cosine_similarity(self.all_data_features[0][all_index_by_class[curr_class]],self.all_data_features[0][temp_probe_index[curr_class]])
          sum_sim = np.sum(sum_sim,axis = 1)
          similarity_list[all_index_by_class[curr_class]] = sum_sim

        similarity_list = np.exp(parameter_sim*similarity_list)
        representative_loss_for_min = np.exp((-parameter_loss)*all_loss)
        representative_loss_for_min_proba = representative_loss_for_min/np.sum(representative_loss_for_min)
        similarity_proba = similarity_list/np.sum(similarity_list)
        if "sim" in FLAGS.extra_name:
          self.all_CL_select_proba = similarity_proba 
        else:
          self.all_CL_select_proba = np.maximum.reduce([representative_loss_for_min_proba , similarity_proba])   
        self.CL_select_proba = self.all_CL_select_proba[train_index]

        print("Average value of clean: ",np.mean(self.all_CL_select_proba[self.clean_index]))
        print("Average value of non-clean: ",np.mean(self.all_CL_select_proba[list(set([i for i in range(len(self.all_data_label))]).difference(set(self.clean_index)))]))
        self.CL_select_proba = self.CL_select_proba/np.sum(self.CL_select_proba)
      else:
        assert (y_train_original[left_train_clean_index] == y_train_clean.squeeze()).all()
        input_tuple_clean = (x_train_clean, y_train_clean.squeeze(),left_train_clean_index)
        self.train_dataflow_clean = self.create_ds_hard_clean(input_tuple_clean, is_train=True,using_generator = False)

    self.val_dataflow = self.create_ds((x_test, y_test.squeeze()),
                                       is_train=False,using_generator = False)
    
    if self.split_probe:
      if not FLAGS.active:
        assert (y_probe == self.all_data_label[probe_index]).all()
        self.probe_dataflow = self.create_ds_hard_clean((x_probe, y_probe.squeeze(),self.probe_index),
                                          is_train=True,using_generator = False)
      else:
        self.probe_dataflow = self.create_ds_hard_clean(self.generator_probe,is_train=True)

    tf.logging.info('Init [{}] dataset loader'.format(self.dataset_name))
    verbose_data('train', x_train, y_train)
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

  def create_ds_hard_clean(self, generator, is_train=True,using_generator = True):
    """Creates tf.data object given data.

    Args:
      data: data in format of tuple, e.g. (data, label)
      is_train: bool indicate train stage the original copy, so the resulting
        tensor is 5D

    Returns:
      An tf.data.Dataset object
    """
    if using_generator:
      ds = tf.data.Dataset.from_generator(generator,(tf.int64,tf.int64,tf.int64),output_shapes = self.output_clean_shapes )
    else:
      ds = tf.data.Dataset.from_tensor_slices(generator)

    map_fn = lambda x, y,z: (cifar_process(x, is_train), y,z)
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
                '''
                for name, module in feature_extractor_1.named_modules():
                    short_name = name[:-1]
                    if short_name.endswith("bn") or short_name.endswith("conv"):
                        all_weight_layer[name] = module.weight
                        print(name)
                        print("-------")
                        print(all_weight_layer[name].grad.size())
                print("End here")
                '''
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
                # curr_feature_1 =  activation_1["layer4"]
                # curr_feature_2 =  activation_2["layer4"]

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
        else:
            with torch.no_grad():
                if FLAGS.AL_model.startswith("BYOL"):
                    curr_feature = feature_extractor(curr_img,return_embedding = True)
                    curr_feature = curr_feature.detach().cpu().numpy().squeeze()
                elif FLAGS.AL_model.startswith("SimCLR"):
                    representation,projection = feature_extractor(curr_img)
                    if FLAGS.projection:
                        curr_feature = projection
                    else:
                        curr_feature = representation
                    curr_feature = curr_feature.detach().cpu().numpy().squeeze()
                elif FLAGS.AL_model.startswith("DivideMix"):
                    prediction_1,curr_feature_1 =  feature_extractor_1(curr_img)
                    prediction_1 = nn.functional.softmax(prediction_1)
                    prediction_2,curr_feature_2 =  feature_extractor_2(curr_img)
                    prediction_2 = nn.functional.softmax(prediction_2)
                    prediction_1 = prediction_1.detach().cpu().numpy().squeeze()
                    prediction_2 = prediction_2.detach().cpu().numpy().squeeze()
                    #curr_feature_1 =  activation_1["layer4"]
                    #curr_feature_2 =  activation_2["layer4"]
                    curr_feature_1 = curr_feature_1.detach().cpu().numpy().squeeze()
                    curr_feature_2 = curr_feature_2.detach().cpu().numpy().squeeze()
                    curr_feature = (curr_feature_1 + curr_feature_2)/2
                    curr_prediction = (prediction_1 + prediction_2)/2
                    assert (np.sum(curr_prediction) - 1.0) < 0.01
                    curr_prediction_label = np.argmax(curr_prediction)
                    prediction.append(get_informativeness(prediction_1,prediction_2))
                    prediction_label.append(curr_prediction_label)
                    prediction_proba.append(curr_prediction)
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

    num = total_img
    trainval_partition = [num - num_val, num_val]
    num_point_per_class = round(FLAGS.num_img*FLAGS.probe_dataset_hold_ratio/num_cluster)
    print("num_point_per_class: ",num_point_per_class)
    print("Separation: ",trainval_partition)
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
    return img[noisy_training_idx], label[noisy_training_idx], img[clean_data_idx_all], label[clean_data_idx_all],clean_label[noisy_training_idx],update_noise_mask, img[left_clean_index], label[left_clean_index],all_clean_index,noisy_training_idx,left_clean_index,clean_data_idx,[all_features,all_features_1,all_features_2],prediction_label,np.mean(all_loss,axis = 0)
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
    

  
    print("Balance of class: ",np.unique(correct_label[:FLAGS.num_img],return_counts = True))
    weight_matrix = np.zeros((len(all_clean_index),len(all_noise_index)))
    all_noise_label = noisy_label[all_noise_index]

    all_features_1_noise = all_features_1[all_noise_index]
    all_features_1_clean = all_features_1[all_clean_index]

    all_features_2_noise = all_features_2[all_noise_index]
    all_features_2_clean = all_features_2[all_clean_index]
    print("Confirming correct noise label: ",np.unique(all_noise_label,return_counts = True))
    print("Confirming correct label in noisy label's place: ",np.unique(correct_label[all_noise_index],return_counts = True))
    
    
    if FLAGS.diverse_and_balance.startswith("only_clean_balance_mix_by_class"):
        print("Select clean sample using :",FLAGS.diverse_and_balance)
        all_selected_represeantatives = []
        
        if FLAGS.diverse_and_balance.find("cosine") != -1:
            #weight_matrix_clean_noisy = cosine_similarity(all_features[all_clean_index],all_features[all_noise_index])
            #print("Shape clean noisy: ",weight_matrix_clean_noisy.shape)
            print("Using cosine")
            if FLAGS.diverse_and_balance.startswith("only_clean_balance_mix_by_class_real_features"):
              print("Using real features.....")
              print("Features shape: ",np.shape(all_features))
              weight_matrix_clean_clean = cosine_similarity(all_features[all_clean_index],all_features[all_clean_index]) 
            else:

              curr_all_features = (all_features_1+all_features_2)/2
              print("Using gradient features.....")
              print("Features shape: ",np.shape(curr_all_features))
              weight_matrix_clean_clean = cosine_similarity(curr_all_features[all_clean_index],curr_all_features[all_clean_index]) 
           
        else:
           
            if FLAGS.diverse_and_balance.startswith("only_clean_balance_mix_by_class_real_features"):
              weight_matrix_clean_clean = np.matmul(all_features[all_clean_index],all_features[all_clean_index].T) 
            
       
       
        
        weight_matrix_clean_clean_abs = np.absolute(weight_matrix_clean_clean)
        
       
        for i in range(weight_matrix_clean_clean.shape[0]):
            weight_matrix_clean_clean[i,i] = 0
      
        include_previous = True
        if FLAGS.diverse_and_balance.find(".") != -1:
          add_diverse = FLAGS.diverse_and_balance.split(".")[-1]
        else:
          add_diverse = ""
        print("Finish ",add_diverse)
        all_clean_label = noisy_label[all_clean_index]
        all_idx_by_class = [np.where(all_clean_label == curr_class)[0] for curr_class in range(num_class)]
        
        all_num_sample_by_class = [num_point_per_class for i in range(num_class)]
        print("All all_num_sample_by_class: ",all_num_sample_by_class)
        for curr_class in range(num_class):
            other_class_index = [i for i in range(weight_matrix_clean_clean.shape[0]) if i not in all_idx_by_class[curr_class] and i not in all_selected_represeantatives]
            
            curr_class_selected_represeantatives = []
            
            print("Confirm that there is negative: ",np.min(weight_matrix_clean_clean))
            while len(curr_class_selected_represeantatives) < all_num_sample_by_class[curr_class]:
                tf.logging.info("Using normal diverse and balance")
                left_index = [i for i in all_idx_by_class[curr_class] if i not in curr_class_selected_represeantatives]
                curr_other_classes_matrix = weight_matrix_clean_clean_abs[other_class_index]
                curr_other_classes_matrix = curr_other_classes_matrix[:,left_index]

                curr_class_matrix = weight_matrix_clean_clean[left_index]
                curr_class_matrix = curr_class_matrix[:,left_index]
                internal_class_sum = np.sum(curr_class_matrix,axis = 0)
                distribution_internal_matrix = curr_class_matrix


                if include_previous:
                    curr_internal_matrix = weight_matrix_clean_clean[curr_class_selected_represeantatives]
                    curr_internal_matrix = curr_internal_matrix[:,left_index]
                    curr_internal_matrix = np.sum(curr_internal_matrix,axis = 0)
                    distribution_internal_matrix += curr_internal_matrix
                    curr_internal_matrix_sum = np.sum(curr_internal_matrix)
                    curr_internal_matrix = curr_internal_matrix_sum - curr_internal_matrix
                    internal_class_sum += curr_internal_matrix
                     
                for i in range(distribution_internal_matrix.shape[0]):
                    distribution_internal_matrix[i,i] = 0
                    distribution_internal_matrix[i,i] = np.sum(distribution_internal_matrix[i])/(distribution_internal_matrix.shape[1]-1)
                std_internal = np.std(distribution_internal_matrix,axis = 1)

                sorted_idx_internal_matrix = np.squeeze(np.argsort(internal_class_sum))
                '''
                print("sorted_idx_internal_matrix: ")
                print(sorted_idx_internal_matrix)
                '''
                order_internal_matrix = np.array([i for i in range(len(left_index))])
                for i in range(len(left_index)):
                    order_internal_matrix[sorted_idx_internal_matrix[i]] = i
                order_internal_matrix = np.array(order_internal_matrix)


                external_class_sum = np.sum(curr_other_classes_matrix,axis = 0)
                distribution_external_matrix = curr_other_classes_matrix.T
                if include_previous:
                    curr_external_matrix = weight_matrix_clean_clean[curr_class_selected_represeantatives]
                    curr_external_matrix = curr_external_matrix[:,other_class_index]
                    distribution_external_matrix += np.sum(curr_external_matrix,axis = 0)
                    curr_external_matrix = np.sum(curr_external_matrix)
                    external_class_sum += curr_external_matrix

                std_external = np.std(distribution_internal_matrix,axis = 1)
                
                sorted_idx_external_matrix = np.squeeze(np.argsort(external_class_sum))
                '''
                print("sorted_idx_external_matrix: ")
                print(sorted_idx_external_matrix)
                '''
                order_external_matrix = np.array([i for i in range(len(left_index))])
                for i in range(len(left_index)):
                    order_external_matrix[sorted_idx_external_matrix[i]] = i
                order_external_matrix = np.array(order_external_matrix)
                #final_candidate_matrix = order_internal_matrix - order_external_matrix
                final_candidate_matrix = order_internal_matrix
                tf.logging.info("add_diverse is: " + str(add_diverse))
                tf.logging.info("Found is: " + str(add_diverse.find("cosine")))
                if add_diverse.find("cosine") != -1 and len(curr_class_selected_represeantatives) > 0:
                    print("Adding cosine diversity")
                    internal_weight_matrix = cosine_similarity(all_features[all_clean_index[left_index]],all_features[all_clean_index[curr_class_selected_represeantatives]])
                   
                    internal_weight_matrix = np.mean(internal_weight_matrix,axis = 1)
                    '''
                    final_candidate_matrix = np.divide(final_candidate_matrix,internal_weight_matrix)
                    final_candidate_matrix *= (-1)
                    '''
                    sorted_idx_internal_matrix = np.squeeze(np.argsort(internal_weight_matrix))
                    order_internal_matrix =  np.array([i for i in range(len(left_index))])
                    for i in range(len(left_index)):
                        order_internal_matrix[sorted_idx_internal_matrix[i]] = i
                    order_internal_matrix = np.array(order_internal_matrix)
                    final_candidate_matrix = final_candidate_matrix - order_internal_matrix
                else:
                  print("2 value: ")
                  print(add_diverse.find("cosine"),"    ",len(curr_class_selected_represeantatives))
                if add_diverse.find("std") != -1:
                    print("Adding std diversity")
                    sorted_idx_std_internal = np.squeeze(np.argsort(std_internal))
                    order_std_internal=  np.array([i for i in range(len(left_index))])
                    for i in range(len(left_index)):
                        order_std_internal[sorted_idx_std_internal[i]] = i
                    order_std_internal = np.array(order_std_internal)

                    sorted_idx_std_external= np.squeeze(np.argsort(std_external))
                    order_std_external=  np.array([i for i in range(len(left_index))])
                    for i in range(len(left_index)):
                        order_std_external[sorted_idx_std_external[i]] = i
                    order_std_external = np.array(order_std_external)


                    final_candidate_matrix = final_candidate_matrix - order_std_internal # - order_std_external

                print()
                select_index = np.argmax(final_candidate_matrix)
                print("Internal: ",order_internal_matrix[select_index],"  ",internal_class_sum[select_index],"   ",max(internal_class_sum),"   ",min(internal_class_sum),"  ",np.mean(internal_class_sum))
                print("External: ",order_external_matrix[select_index],"  ",external_class_sum[select_index],"   ",max(external_class_sum),"   ",min(external_class_sum),"  ",np.mean(external_class_sum))
                if add_diverse.find("std") != -1:
                    print("std_internal: ",order_std_internal[select_index],"  ",std_internal[select_index],"   ",max(std_internal),"   ",min(std_internal),"  ",np.mean(std_internal))
                    #print("std_external: ",order_std_external[select_index],"  ",std_external[select_index],"   ",max(std_external),"   ",min(std_external),"  ",np.mean(std_external))
                curr_class_selected_represeantatives.append(left_index[select_index])
            all_selected_represeantatives.extend(curr_class_selected_represeantatives)
        return all_clean_index[all_selected_represeantatives],all_clean_index













                       



