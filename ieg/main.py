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
      use_pretrained = "pretrained"
    else:
      use_pretrained = ""
    if FLAGS.warmup_iteration > 0:
      use_warmup = "_warmup_"+str(FLAGS.warmup_iteration)
    else:
      use_warmup = ""
    if FLAGS.use_GMM_pseudo_classification:
      use_GMM_pseudo_classification = "_use_GMM"
    else:
      use_GMM_pseudo_classification = ""
    if len(FLAGS.update_probe) > 0:
      update_probe = "_"+str(FLAGS.update_probe)+"_thre_"+str(FLAGS.threshold_relabel)

    else:
      update_probe = ""
    if len(FLAGS.using_loss) > 0:
      using_loss = "_"+str(FLAGS.using_loss)

    else:
      using_loss = ""
    FLAGS.checkpoint_path = os.path.join(
        FLAGS.checkpoint_path, '{}_p{}_{}_{}_{}_{}{}{}{}{}'.format(FLAGS.dataset,
                                               FLAGS.probe_dataset_hold_ratio,FLAGS.diverse_and_balance,FLAGS.extra_name,FLAGS.seed,use_pretrained,use_warmup,use_GMM_pseudo_classification,update_probe,using_loss),
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
