# coding=utf-8
"""Main function for the project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

from ieg import options
from ieg.dataset_utils import datasets
from ieg.models.basemodel import BaseModel
from ieg.models.l2rmodel import L2R
from ieg.models.model import IEG

import tensorflow.compat.v1 as tf

logger = tf.get_logger()
logger.propagate = False

FLAGS = flags.FLAGS

options.define_basic_flags()

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def train(model, sess):
  """Training launch function."""
  with sess.as_default():
    model.train()


def evaluation(model, sess):
  """Evaluation launch function."""
  with sess.as_default():
    model.evaluation()


def main(_):
  tf.disable_v2_behavior()

  strategy = tf.distribute.MirroredStrategy()
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  # Creates dataset
  if 'cifar' in FLAGS.dataset:
    dataset = datasets.CIFAR()
    if FLAGS.pretrained_noise:
      use_pretrained = "_pretrained"
    else:
      use_pretrained = ""

    if FLAGS.warmup_iteration > 0:
      use_warmup = "_warm_"+str(FLAGS.warmup_iteration)
    else:
      use_warmup = ""

    if len(FLAGS.update_probe) > 0:
      update_probe = "_"+str(FLAGS.update_probe)+"_thre_"+str(FLAGS.threshold_relabel)+"_"+str(FLAGS.threshold_clean)
    else:
      update_probe = ""

    if len(FLAGS.using_loss) > 0:
      using_loss = "_"+str(FLAGS.using_loss)
    else:
      using_loss = ""

    if FLAGS.scaled_loss != 1:
      scaled_loss = "_sca_l_"+str(FLAGS.scaled_loss)
    else:
      scaled_loss = ""

    if "_" in FLAGS.update_loss:
      params = "_"+str(FLAGS.update_loss)
    else:
      params = ""
    if len(FLAGS.using_new_features) > 0:
      new = "_"+str(FLAGS.using_new_features)
    else:
      new = ""
    FLAGS.checkpoint_path = os.path.join(
        FLAGS.checkpoint_path, '{}_p{}_{}{}_{}{}{}{}{}{}{}'.format(FLAGS.dataset,
                                               FLAGS.probe_dataset_hold_ratio,FLAGS.seed,use_pretrained,FLAGS.extra_name,use_warmup,update_probe,using_loss,scaled_loss,params,new),
        FLAGS.network_name)
  elif 'webvisionmini' in FLAGS.dataset:
    # webvision mini version
    dataset = datasets.WebVision(
        root='./ieg/data/tensorflow_datasets/',
        version='webvisionmini-google',
        use_imagenet_as_eval=FLAGS.use_imagenet_as_eval)
    FLAGS.checkpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.dataset,
                                         FLAGS.network_name)

  if FLAGS.method == 'supervised':
    model = BaseModel(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'l2r':
    model = L2R(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'ieg':
    model = IEG(sess=sess, strategy=strategy, dataset=dataset)
  else:
    raise NotImplementedError('{} is not existed'.format(FLAGS.method))

  if FLAGS.mode == 'evaluation':
    evaluation(model, sess)
  else:
    train(model, sess)

if __name__ == '__main__':
  app.run(main)
