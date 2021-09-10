# coding=utf-8
"""The proposed model training code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg import utils
from ieg.dataset_utils.utils import autoaug_batch_process_map_fn,autoaug_batch_process_map_reset_fn
from ieg.models import networks
from ieg.models.basemodel import BaseModel
from ieg.models.custom_ops import logit_norm
from ieg.models.custom_ops import MixMode
from ieg.dataset_utils.datasets import upload_checkpoint 
import torch.nn.functional as functional
import torch

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import shutil 

FLAGS = flags.FLAGS
logging = tf.logging


class IEG(BaseModel):
  """Model training class."""

  def __init__(self, sess, strategy, dataset):
    super(IEG, self).__init__(sess, strategy, dataset)
    logging.info('Init IEG model')

    self.augment = MixMode()
    self.beta = 0.5  # MixUp hyperparam
    self.nu = 2      # K value for label guessing

  def set_input(self):
    if len(self.dataset.train_dataflow.output_shapes[0]) == 3:
      # Use for cifar
      if not FLAGS.active:
        train_ds = self.dataset.train_dataflow.shuffle(
            buffer_size=self.batch_size * 10).repeat().batch(
                self.batch_size, drop_remainder=True
            ).map(
                # strong augment each batch data and expand to 5D [Bx2xHxWx3]
                autoaug_batch_process_map_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
      else:
        train_ds = self.dataset.train_dataflow.batch(
              self.batch_size, drop_remainder=True
          ).map(
              # strong augment each batch data and expand to 5D [Bx2xHxWx3]
              autoaug_batch_process_map_fn,
              num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                  buffer_size=tf.data.experimental.AUTOTUNE)
        train_reset_ds = self.dataset.train_dataflow_reset.batch(
              self.batch_size, drop_remainder=False
          ).map(
              # strong augment each batch data and expand to 5D [Bx2xHxWx3]
              autoaug_batch_process_map_reset_fn,
              num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                  buffer_size=tf.data.experimental.AUTOTUNE)
        train_clean_ds = self.dataset.train_dataflow_clean.shuffle(
            buffer_size=self.batch_size * 10).repeat().batch(
                self.batch_size, drop_remainder=True
            ).map(
                # strong augment each batch data and expand to 5D [Bx2xHxWx3]
                autoaug_batch_process_map_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
    else:
      if not FLAGS.active:
        train_ds = self.dataset.train_dataflow.shuffle(
            buffer_size=self.batch_size * 10).repeat().batch(
                self.batch_size, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
      else:
        train_ds = self.dataset.train_dataflow.batch(
              self.batch_size, drop_remainder=True).prefetch(
                  buffer_size=tf.data.experimental.AUTOTUNE)
        train_reset_ds = self.dataset.train_dataflow_reset.batch(
                  self.batch_size, drop_remainder=False).prefetch(
                      buffer_size=tf.data.experimental.AUTOTUNE)
      

        train_clean_ds = self.dataset.train_dataflow_clean.shuffle(
            buffer_size=self.batch_size * 10).repeat().batch(
                self.batch_size, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
    # no shuffle for probe, so a batch is class balanced.
    probe_ds = self.dataset.probe_dataflow.repeat().batch(
        self.batch_size, drop_remainder=True).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = self.dataset.val_dataflow.batch(
        FLAGS.val_batch_size, drop_remainder=False).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    self.train_input_iterator = (
        self.strategy.experimental_distribute_dataset(
            train_ds).make_initializable_iterator())
    self.probe_input_iterator = (
        self.strategy.experimental_distribute_dataset(
            probe_ds).make_initializable_iterator())

    if FLAGS.active:
      self.train_clean_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              train_clean_ds).make_initializable_iterator())
      self.train_reset_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              train_reset_ds).make_initializable_iterator())

    self.eval_input_iterator = (
        self.strategy.experimental_distribute_dataset(
            val_ds).make_initializable_iterator())

  def meta_momentum_update(self, grad, var_name, optimizer):
    # Finds corresponding momentum of a var name
    accumulation = utils.get_var(optimizer.variables(), var_name.split(':')[0])
    if len(accumulation) != 1:
      raise ValueError('length of accumulation {}'.format(len(accumulation)))
    new_grad = tf.math.add(
        tf.stop_gradient(accumulation[0]) * FLAGS.meta_momentum, grad)
    return new_grad

  def guess_label(self, logit, temp=0.5):
    logit = tf.reshape(logit, [-1, self.dataset.num_classes])
    logit = tf.split(logit, self.nu, axis=0)
    logit = [logit_norm(x) for x in logit]
    logit = tf.concat(logit, 0)
    ## Done with logit norm
    p_model_y = tf.reshape(
        tf.nn.softmax(logit), [self.nu, -1, self.dataset.num_classes])
    p_model_y = tf.reduce_mean(p_model_y, axis=0)

    p_target = tf.pow(p_model_y, 1.0 / temp)
    p_target /= tf.reduce_sum(p_target, axis=1, keepdims=True)

    return p_target

  def crossentropy_minimize(self,
                            u_logits,
                            u_images,
                            l_images,
                            l_labels,
                            u_labels=None):
    """Cross-entropy optimization step implementation for TPU."""
    batch_size = self.batch_size // self.strategy.num_replicas_in_sync
    guessed_label = self.guess_label(u_logits)
    self.guessed_label = guessed_label

    guessed_label = tf.reshape(
        tf.stop_gradient(guessed_label), shape=(-1, self.dataset.num_classes))

    l_labels = tf.reshape(
        tf.one_hot(l_labels, self.dataset.num_classes),
        shape=(-1, self.dataset.num_classes))
    augment_images, augment_labels = self.augment(
        [l_images, u_images], [l_labels] + [guessed_label] * self.nu,
        [self.beta, self.beta])
    logit,_ = self.net(augment_images, name='model', training=True)

    zbs = batch_size * 2
    halfzbs = batch_size

    split_pos = [tf.shape(l_images)[0], halfzbs, halfzbs]

    logit = [logit_norm(lgt) for lgt in tf.split(logit, split_pos, axis=0)]
    u_logit = tf.concat(logit[1:], axis=0)

    split_pos = [tf.shape(l_images)[0], zbs]
    l_augment_labels, u_augment_labels = tf.split(
        augment_labels, split_pos, axis=0)

    u_loss = tf.losses.softmax_cross_entropy(u_augment_labels, u_logit)
    l_loss = tf.losses.softmax_cross_entropy(l_augment_labels, logit[0])

    loss = tf.math.add(
        l_loss, u_loss * FLAGS.ce_factor, name='crossentropy_minimization_loss')

    return loss

  def consistency_loss(self, logit, aug_logit):

    def kl_divergence(q_logits, p_logits):
      q = tf.nn.softmax(q_logits)
      per_example_kl_loss = q * (
          tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits))
      return tf.reduce_mean(per_example_kl_loss) * self.dataset.num_classes

    return tf.math.multiply(
        kl_divergence(tf.stop_gradient(logit), aug_logit),
        FLAGS.consistency_factor,
        name='consistency_loss')

  def unsupervised_loss(self):
    """Creates unsupervised losses.

    Here we create two cross-entropy losses and a KL-loss defined in the paper.

    Returns:
      A list of losses.
    """

    if FLAGS.ce_factor == 0 and FLAGS.consistency_factor == 0:
      return [tf.constant(0, tf.float32), tf.constant(0, tf.float32)]
    logits = self.logits
    images = self.images
    aug_images = self.aug_images
    probe_images, probe_labels = self.probe_images, self.probe_labels
    im_shape = (-1, int(probe_images.shape[1]), int(probe_images.shape[2]),
                int(probe_images.shape[3]))

    aug_logits,_ = self.net(aug_images, name='model', training=True)

    n_probe_to_mix = tf.shape(aug_images)[0]
    probe = tf.tile(tf.constant([[10.]]), [1, tf.shape(probe_images)[0]])
    idx = tf.squeeze(tf.random.categorical(probe, n_probe_to_mix))

    l_images = tf.reshape(tf.gather(probe_images, idx), im_shape)
    l_labels = tf.reshape(tf.gather(probe_labels, idx), (-1,))

    u_logits = tf.concat([logits, aug_logits], axis=0)
    u_images = tf.concat([images, aug_images], axis=0)

    losses = []
    if FLAGS.ce_factor > 0:
      logging.info('Use crossentropy minimization loss {}'.format(
          FLAGS.ce_factor))
      ce_min_loss = self.crossentropy_minimize(u_logits, u_images, l_images,
                                               l_labels)
      losses.append(ce_min_loss)
    else:
      losses.append(tf.constant(0, tf.float32))

    if FLAGS.consistency_factor > 0:
      logging.info('Use consistency loss {}'.format(
          FLAGS.consistency_factor))
      consis_loss = self.consistency_loss(logits, aug_logits)
      losses.append(consis_loss)

    else:
      losses.append(tf.constant(0, tf.float32))

    return losses,aug_logits

  def meta_optimize(self):
    """Meta optimization step."""

    probe_images, probe_labels = self.probe_images, self.probe_labels
    labels = self.labels
    net = self.net
    logits = self.logits
    gate_gradients = 1

    batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)
    init_eps_val = float(1) / batch_size

    meta_net = networks.MetaImage(self.net, name='meta_model')

    if FLAGS.meta_momentum and not self.optimizer.variables():
      # Initializing momentum state of optimizer for meta momentum update.
      # It is a hacky implementation
      logging.info('Pre-initialize optimizer momentum states.')
      idle_net_cost = tf.losses.sparse_softmax_cross_entropy(
          self.labels, logits)
      tmp_var_grads = self.optimizer.compute_gradients(
          tf.reduce_mean(idle_net_cost), net.trainable_variables)
      self.optimizer.apply_gradients(tmp_var_grads)

    with tf.name_scope('coefficient'):
      # Data weight coefficient
      target = tf.constant(
          [init_eps_val] * batch_size,
          shape=(batch_size,),
          dtype=np.float32,
          name='weight')
      # Data re-labeling coefficient
      eps = tf.constant(
          [FLAGS.grad_eps_init] * batch_size,
          shape=(batch_size,),
          dtype=tf.float32,
          name='eps')

    onehot_labels = tf.one_hot(labels, self.dataset.num_classes)
    onehot_labels = tf.cast(onehot_labels, tf.float32)
    eps_k = tf.reshape(eps, [batch_size, 1])

    mixed_labels = eps_k * onehot_labels + (1 - eps_k) * self.guessed_label
    # raw softmax loss
    log_softmax = tf.nn.log_softmax(logits)
    net_cost = -tf.reduce_sum(mixed_labels * log_softmax, 1)

    lookahead_loss = tf.reduce_sum(tf.multiply(target, net_cost))
    lookahead_loss = lookahead_loss + net.regularization_loss

    with tf.control_dependencies([lookahead_loss]):
      train_vars = net.trainable_variables
      var_grads = tf.gradients(
          lookahead_loss, train_vars, gate_gradients=gate_gradients)

      static_vars = []
      for i in range(len(train_vars)):
        if FLAGS.meta_momentum > 0:
          actual_grad = self.meta_momentum_update(var_grads[i],
                                                  train_vars[i].name,
                                                  self.optimizer)
          static_vars.append(
              tf.math.subtract(train_vars[i],
                               FLAGS.meta_stepsize * actual_grad))
        else:
          static_vars.append(
              tf.math.subtract(train_vars[i],
                               FLAGS.meta_stepsize * var_grads[i]))
        # new style
        meta_net.add_variable_alias(
            static_vars[-1], var_name=train_vars[i].name)

      for uv in net.updates_variables:
        meta_net.add_variable_alias(
            uv, var_name=uv.name, var_type='updates_variables')
      meta_net.verbose()

    with tf.control_dependencies(static_vars):
      g_logits,_ = meta_net(
          probe_images, name='meta_model', reuse=True, training=True)

      desired_y = tf.one_hot(probe_labels, self.dataset.num_classes)
      meta_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
          desired_y, g_logits)
      meta_loss = tf.reduce_mean(meta_loss, name='meta_loss')
      meta_loss = meta_loss + meta_net.get_regularization_loss(net.wd)
      meta_acc, meta_acc_op = tf.metrics.accuracy(probe_labels,
                                                  tf.argmax(g_logits, axis=1))

    with tf.control_dependencies([meta_loss] + [meta_acc_op]):
      meta_train_vars = meta_net.trainable_variables
      grad_meta_vars = tf.gradients(
          meta_loss, meta_train_vars, gate_gradients=gate_gradients)
      grad_target, grad_eps = tf.gradients(
          static_vars, [target, eps],
          grad_ys=grad_meta_vars,
          gate_gradients=gate_gradients)
    # updates weight
    raw_weight = target - grad_target
    raw_weight = raw_weight - init_eps_val
    unorm_weight = tf.clip_by_value(
        raw_weight, clip_value_min=0, clip_value_max=float('inf'))
    norm_c = tf.reduce_sum(unorm_weight)
    weight = tf.divide(unorm_weight, norm_c + 0.00001)

    # gets new lambda by the sign of gradient
    new_eps = tf.where(grad_eps < 0, x=tf.ones_like(eps), y=tf.zeros_like(eps))

    return tf.stop_gradient(weight), tf.stop_gradient(
        new_eps), meta_loss, meta_acc, tf.stop_gradient(g_logits)

  def train_step(self):

    def step_fn(inputs):
      """Step functon.

      Args:
        inputs: inputs from data iterator

      Returns:
        a set of variables want to observe in Tensorboard
      """
      net = self.net
      (all_images, labels,sample_index), (self.probe_images, self.probe_labels,probe_index) = inputs
      sample_index = tf.cast(sample_index,tf.int32)

      images, self.aug_images = all_images[:, 0], all_images[:, 1]

      self.images, self.labels = images, labels
      batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)

      logits,_ = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      self.logits = logits

      # other losses
      # initialized first to use self.guessed_label for meta step
      (xe_loss, cs_loss),aug_logits = self.unsupervised_loss()
      logits_softmax = tf.stop_gradient(tf.nn.softmax(logits))
      aug_logits_softmax = tf.stop_gradient(tf.nn.softmax(aug_logits))
      final_logit_softmax_2 = tf.stop_gradient(tf.reduce_mean([logits_softmax,aug_logits_softmax],axis = 0))

      # meta optimization
      weight, eps, meta_loss, meta_acc, g_logits_softmax = self.meta_optimize()
      g_logits_softmax_2 = tf.stop_gradient(tf.nn.softmax(g_logits_softmax))

      pseudo_labels = tf.argmax(logits,axis =1)
      pseudo_labels_probe = tf.argmax(g_logits_softmax,axis =1)

      ## losses w.r.t new weight and loss
      onehot_labels = tf.one_hot(labels, self.dataset.num_classes)
      onehot_labels = tf.cast(onehot_labels, tf.float32)
      eps_k = tf.reshape(eps, [batch_size, 1])

      mixed_labels = tf.math.add(
          eps_k * onehot_labels, (1 - eps_k) * self.guessed_label,
          name='mixed_labels')
      net_cost = tf.losses.softmax_cross_entropy(
          mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      # loss with initial weight
      net_loss1 = tf.reduce_mean(net_cost)

      # loss with initial eps
      init_eps = tf.constant(
          [FLAGS.grad_eps_init] * batch_size, dtype=tf.float32)
      init_eps = tf.reshape(init_eps, (-1, 1))
      init_mixed_labels = tf.math.add(
          init_eps * onehot_labels, (1 - init_eps) * self.guessed_label,
          name='init_mixed_labels')

      net_cost2 = tf.losses.softmax_cross_entropy(
          init_mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      net_loss2 = tf.reduce_sum(tf.math.multiply(net_cost2, weight))

      net_loss = (net_loss1 + net_loss2) / 2
      if FLAGS.scaled_loss >1:
        net_loss *= FLAGS.scaled_loss
        net_loss = net_loss + tf.add_n([xe_loss, cs_loss])
      else:
        net_loss = net_loss + tf.add_n([xe_loss, cs_loss])*FLAGS.scaled_loss
      net_loss += net.regularization_loss
      net_loss /= self.strategy.num_replicas_in_sync

      # rescale by gpus
      with tf.control_dependencies(net.updates):
        net_grads = tf.gradients(net_loss, net.trainable_variables)
        minimizer_op = self.optimizer.apply_gradients(
            zip(net_grads, net.trainable_variables),
            global_step=self.global_step)

      with tf.control_dependencies([minimizer_op]):
        train_op = self.ema.apply(net.trainable_variables)

      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))

      with tf.control_dependencies([train_op, acc_update_op]):
        return (tf.identity(net_loss), tf.identity(xe_loss),
                tf.identity(cs_loss), tf.identity(meta_loss),
                tf.identity(meta_acc), tf.identity(acc_op), tf.identity(weight),
                tf.identity(labels),tf.identity(sample_index),tf.identity(probe_index),tf.identity(pseudo_labels),tf.identity(pseudo_labels_probe),tf.identity(final_logit_softmax_2),tf.identity(g_logits_softmax_2))

    # end of parallel
    (pr_net_loss, pr_xe_loss, pr_cs_loss, pr_metaloss, pr_metaacc, pr_acc,
     pr_weight, pr_labels,sample_index,probe_index,pseudo_labels,pseudo_labels_probe,final_logit_softmax,final_logit_softmax_probe) = self.strategy.run(
         step_fn,
         args=((next(self.train_input_iterator),
                next(self.probe_input_iterator)),))
    # collect device variables
    weights = self.strategy.unwrap(pr_weight)
    weights = tf.concat(weights, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)

    pseudo_labels = self.strategy.unwrap(pseudo_labels)
    pseudo_labels = tf.concat(pseudo_labels, axis=0)
    pseudo_labels_probe = self.strategy.unwrap(pseudo_labels_probe)
    pseudo_labels_probe = tf.concat(pseudo_labels_probe, axis=0)
    final_logit_softmax = self.strategy.unwrap(final_logit_softmax)
    final_logit_softmax = tf.concat(final_logit_softmax, axis=0)
    final_logit_softmax_probe = self.strategy.unwrap(final_logit_softmax_probe)
    final_logit_softmax_probe = tf.concat(final_logit_softmax_probe, axis=0)

   
    
    sample_index = self.strategy.unwrap(sample_index)
    sample_index = tf.concat(sample_index, axis=0)
    probe_index = self.strategy.unwrap(probe_index)
    probe_index = tf.concat(probe_index, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_metaacc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaacc)
    net_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_net_loss)
    xe_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_xe_loss)
    cs_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_cs_loss)
    meta_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaloss)

    # The following add variables for tensorboard visualization
    merges = []
    merges.append(tf.summary.scalar('acc/train', mean_acc))
    merges.append(tf.summary.scalar('loss/xemin', xe_loss))
    merges.append(tf.summary.scalar('loss/consistency', cs_loss))
    merges.append(tf.summary.scalar('loss/net', net_loss))
    merges.append(tf.summary.scalar('loss/meta', meta_loss))
    merges.append(tf.summary.scalar('acc/meta', mean_metaacc))
    merges.append(
        tf.summary.scalar('acc/eval_on_train', self.eval_acc_on_train[0]))
    merges.append(
        tf.summary.scalar('acc/eval_on_train_top5', self.eval_acc_on_train[1]))
    merges.append(tf.summary.scalar('acc/num_eval', self.eval_acc_on_train[2]))

    zw_inds = tf.squeeze(
        tf.where(tf.less_equal(weights, 0), name='zero_weight_index'))
    merges.append(
        tf.summary.scalar(
            'weights/zeroratio',
            tf.math.divide(
                tf.cast(tf.size(zw_inds), tf.float32),
                tf.cast(tf.size(weights), tf.float32))))

    self.epoch_var = tf.cast(
        self.global_step / self.iter_epoch, tf.float32, name='epoch')
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    summary = tf.summary.merge(merges)

    return [
        net_loss, meta_loss, xe_loss, cs_loss, mean_acc, mean_metaacc, summary,
        weights,sample_index,probe_index,pseudo_labels,pseudo_labels_probe,final_logit_softmax,final_logit_softmax_probe
    ]



  def train_step_clean(self):

    def step_fn_clean(inputs):
      """Step functon.

      Args:
        inputs: inputs from data iterator

      Returns:
        a set of variables want to observe in Tensorboard
      """
      net = self.net
      (all_images, labels,sample_index), (self.probe_images, self.probe_labels,self.probe_index) = inputs
      assert len(all_images.shape) == 5
      images, self.aug_images = all_images[:, 0], all_images[:, 1]

      self.images, self.labels = images, labels
      batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)

      logits,_ = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      self.logits = logits

      # other losses
      # initialized first to use self.guessed_label for meta step
      (xe_loss, cs_loss),aug_logits = self.unsupervised_loss()
      logits_softmax = tf.stop_gradient(tf.nn.softmax(logits))
      aug_logits_softmax = tf.stop_gradient(tf.nn.softmax(aug_logits))
      final_logit_softmax_2 = tf.stop_gradient(tf.reduce_mean([logits_softmax,aug_logits_softmax],axis = 0))

      # meta optimization
      weight, eps, meta_loss, meta_acc, g_logits_softmax = self.meta_optimize()
      g_logits_softmax_2 = tf.stop_gradient(tf.nn.softmax(g_logits_softmax))

      ## losses w.r.t new weight and loss
      onehot_labels = tf.one_hot(labels, self.dataset.num_classes)
      onehot_labels = tf.cast(onehot_labels, tf.float32)
      eps_k = tf.reshape(eps, [batch_size, 1])

      mixed_labels = tf.math.add(
          eps_k * onehot_labels, (1 - eps_k) * self.guessed_label,
          name='mixed_labels')
      net_cost = tf.losses.softmax_cross_entropy(
          mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      # loss with initial weight
      net_loss1 = tf.reduce_mean(net_cost)

      # loss with initial eps
      init_eps = tf.constant(
          [FLAGS.grad_eps_init] * batch_size, dtype=tf.float32)
      init_eps = tf.reshape(init_eps, (-1, 1))
      init_mixed_labels = tf.math.add(
          init_eps * onehot_labels, (1 - init_eps) * self.guessed_label,
          name='init_mixed_labels')

      net_cost2 = tf.losses.softmax_cross_entropy(
          init_mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      net_loss2 = tf.reduce_sum(tf.math.multiply(net_cost2, weight))

      net_loss = (net_loss1 + net_loss2) / 2
      net_loss *= FLAGS.scaled_loss_clean
      net_loss = net_loss + tf.add_n([xe_loss, cs_loss])
      net_loss += net.regularization_loss
      net_loss /= self.strategy.num_replicas_in_sync

      # rescale by gpus
      with tf.control_dependencies(net.updates):
        net_grads = tf.gradients(net_loss, net.trainable_variables)
        minimizer_op = self.optimizer.apply_gradients(
            zip(net_grads, net.trainable_variables),
            global_step=self.global_step)

      with tf.control_dependencies([minimizer_op]):
        train_op = self.ema.apply(net.trainable_variables)

      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))

      with tf.control_dependencies([train_op, acc_update_op]):
        return (tf.identity(net_loss), tf.identity(xe_loss),
                tf.identity(cs_loss), tf.identity(meta_loss),
                tf.identity(meta_acc), tf.identity(acc_op), tf.identity(weight),
                tf.identity(labels),tf.identity(sample_index),tf.identity(self.probe_index),tf.identity(final_logit_softmax_2),tf.identity(g_logits_softmax_2))
    # end of parallel
    (pr_net_loss, pr_xe_loss, pr_cs_loss, pr_metaloss, pr_metaacc, pr_acc,
     pr_weight, pr_labels,sample_index,probe_index,final_logit_softmax,final_logit_softmax_probe) = self.strategy.run(
         step_fn_clean,
         args=((next(self.train_clean_input_iterator),
                next(self.probe_input_iterator)),))
    # collect device variables
    weights = self.strategy.unwrap(pr_weight)
    weights = tf.concat(weights, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)


    final_logit_softmax = self.strategy.unwrap(final_logit_softmax)
    final_logit_softmax = tf.concat(final_logit_softmax, axis=0)
    final_logit_softmax_probe = self.strategy.unwrap(final_logit_softmax_probe)
    final_logit_softmax_probe = tf.concat(final_logit_softmax_probe, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_metaacc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaacc)
    net_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_net_loss)
    xe_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_xe_loss)
    cs_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_cs_loss)
    meta_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaloss)

    # The following add variables for tensorboard visualization
    merges = []
    merges.append(tf.summary.scalar('acc/train', mean_acc))
    merges.append(tf.summary.scalar('loss/xemin', xe_loss))
    merges.append(tf.summary.scalar('loss/consistency', cs_loss))
    merges.append(tf.summary.scalar('loss/net', net_loss))
    merges.append(tf.summary.scalar('loss/meta', meta_loss))
    merges.append(tf.summary.scalar('acc/meta', mean_metaacc))
    merges.append(
        tf.summary.scalar('acc/eval_on_train', self.eval_acc_on_train[0]))
    merges.append(
        tf.summary.scalar('acc/eval_on_train_top5', self.eval_acc_on_train[1]))
    merges.append(tf.summary.scalar('acc/num_eval', self.eval_acc_on_train[2]))

    zw_inds = tf.squeeze(
        tf.where(tf.less_equal(weights, 0), name='zero_weight_index'))
    merges.append(
        tf.summary.scalar(
            'weights/zeroratio',
            tf.math.divide(
                tf.cast(tf.size(zw_inds), tf.float32),
                tf.cast(tf.size(weights), tf.float32))))

    self.epoch_var = tf.cast(
        self.global_step / self.iter_epoch, tf.float32, name='epoch')
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    summary = tf.summary.merge(merges)

    return [
        net_loss, meta_loss, xe_loss, cs_loss, mean_acc, mean_metaacc, summary,
        weights,sample_index,probe_index,final_logit_softmax,final_logit_softmax_probe
    ]


  def train(self):
    self.set_input()
    list_update_probe = []
    list_update_clean = []
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()

      if not FLAGS.active:
        self.sess.run([
          self.train_input_iterator.initializer,
          self.probe_input_iterator.initializer
        ])
      else:
        self.sess.run([
            self.train_input_iterator.initializer,
            self.probe_input_iterator.initializer,
            self.train_clean_input_iterator.initializer,
        ])
      if len(FLAGS.update_probe) > 0 :
        self.sess.run([
            self.train_reset_input_iterator.initializer
        ])
      self.sess.run([self.eval_input_iterator.initializer])

      logging.info('Finish variable initialization')
      iter_epoch = self.iter_epoch

      self.saver = tf.train.Saver(max_to_keep=4)
      if len(FLAGS.pretrained_path) > 0 or FLAGS.using_colab:
        shutil.rmtree(FLAGS.checkpoint_path)
        shutil.copytree("data-ieg-active/"+str(FLAGS.checkpoint_path.split("/")[-1]),FLAGS.checkpoint_path) 

      self.load_model()
      FLAGS.restore_step = self.global_step.eval()

      pbar = tqdm(total=(FLAGS.max_iteration - FLAGS.restore_step))

      update_probe_interval = 500000
      update_clean_interval = 500000
      probe_lr_threshold = -1
      clean_lr_threshold = -1
    
      if "-" in FLAGS.update_probe:
        update_probe_interval = int(FLAGS.update_probe.split("-")[-1].split("_")[-1])
        probe_lr_threshold = float(FLAGS.update_probe.split("-")[-1].split("_")[-2])
        update_clean_interval = int(FLAGS.update_probe.split("-")[-2].split("_")[-1])
        clean_lr_threshold = float(FLAGS.update_probe.split("-")[-2].split("_")[-2])

      for iteration in range(FLAGS.restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        if iteration < FLAGS.warmup_iteration:
          assert FLAGS.active == True
          curr_train_op = self.train_clean_op
          (lr, net_loss, meta_loss, xe_loss, cs_loss, acc, meta_acc,
           merged_summary, weights,sample_index,probe_index,final_logit_softmax,final_logit_softmax_probe) = (
               self.sess.run([self.learning_rate] + curr_train_op))

          
          if FLAGS.update_probe:
            curr_samples_logits = self.dataset.all_predict_logit[0][sample_index][:,1:,:].copy()
            self.dataset.all_predict_logit[0][sample_index,:-1,:] = curr_samples_logits
            self.dataset.all_predict_logit[0][sample_index,-1,:] = final_logit_softmax
            self.dataset.all_predict_logit[1][sample_index] += 1
            self.dataset.all_predict_logit[1][sample_index] = (self.dataset.all_predict_logit[1][sample_index] <= 5)*self.dataset.all_predict_logit[1][sample_index] + (self.dataset.all_predict_logit[1][sample_index] > 5)*5
            
            curr_samples_logits_probe = self.dataset.all_predict_logit[0][probe_index][:,1:].copy()
            self.dataset.all_predict_logit[0][probe_index,:-1,:] = curr_samples_logits_probe
            self.dataset.all_predict_logit[0][probe_index,-1,:] = final_logit_softmax_probe            
            self.dataset.all_predict_logit[1][probe_index] += 1
            self.dataset.all_predict_logit[1][probe_index] = (self.dataset.all_predict_logit[1][probe_index] <= 5)*self.dataset.all_predict_logit[1][probe_index] + (self.dataset.all_predict_logit[1][probe_index] > 5)*5
        else:
          curr_train_op = self.train_op
          (lr, net_loss, meta_loss, xe_loss, cs_loss, acc, meta_acc,
           merged_summary, weights,sample_index,probe_index,pseudo_labels,pseudo_label_probe,final_logit_softmax,final_logit_softmax_probe) = (
               self.sess.run([self.learning_rate] + curr_train_op))

          
          if FLAGS.update_probe:
            curr_samples_logits = self.dataset.all_predict_logit[0][sample_index][:,1:,:].copy()
            self.dataset.all_predict_logit[0][sample_index,:-1,:] = curr_samples_logits
            self.dataset.all_predict_logit[0][sample_index,-1,:] = final_logit_softmax
            self.dataset.all_predict_logit[1][sample_index] += 1
            self.dataset.all_predict_logit[1][sample_index] = (self.dataset.all_predict_logit[1][sample_index] <= 5)*self.dataset.all_predict_logit[1][sample_index] + (self.dataset.all_predict_logit[1][sample_index] > 5)*5

            curr_samples_logits_probe = self.dataset.all_predict_logit[0][probe_index][:,1:].copy()
            self.dataset.all_predict_logit[0][probe_index,:-1,:] = curr_samples_logits_probe
            self.dataset.all_predict_logit[0][probe_index,-1,:] = final_logit_softmax_probe            
            self.dataset.all_predict_logit[1][probe_index] += 1
            self.dataset.all_predict_logit[1][probe_index] = (self.dataset.all_predict_logit[1][probe_index] <= 5)*self.dataset.all_predict_logit[1][probe_index] + (self.dataset.all_predict_logit[1][probe_index] > 5)*5
           
            
        pbar.update(1)
        message = ('Epoch {}[{}/{}] lr{:.3f} meta_loss:{:.2f} loss:{:.2f} '
                   'mc_loss:{:.2f} uc_loss:{:.2f} weight{:.2f}({:.2f}) '
                   'acc:{:.2f} mata_acc{:.2f}  clean_size{:.2f} train_size{:.2f} acc_real{:.2f} acc_cl{:.2f}').format(iteration // iter_epoch,
                                                       iteration % iter_epoch,
                                                       iter_epoch, lr,
                                                       float(meta_loss),
                                                       float(net_loss),
                                                       float(xe_loss),
                                                       float(cs_loss),
                                                       float(np.mean(weights)),
                                                       float(np.std(weights)),
                                                       float(acc),
                                                       float(meta_acc),float(len(self.dataset.clean_index)),self.dataset.train_data.shape[0],self.dataset.real_acc_train,self.dataset.acc_clean)
        pbar.set_description(message)
        self.summary_writer.add_summary(merged_summary, iteration)
        update_probe_now = False
        update_clean_now = False
        if FLAGS.update_probe != "":
          if iteration > FLAGS.warmup_iteration:
            if len(self.dataset.noise_index) > 0 and ((lr <= clean_lr_threshold and iteration % update_clean_interval == 1) or lr == 0):
                update_clean_now = True
                list_update_clean.append(iteration)
            if (lr <= probe_lr_threshold and iteration % update_probe_interval == 1 and len(FLAGS.update_probe) > 0) or lr == 0:
                update_probe_now = True
                list_update_probe.append(iteration)

            if update_clean_now or update_probe_now:
                print("Active selection at iteration ",iteration," with update_clean_now ",update_clean_now,"  and update_probe_now ",update_probe_now)
                print("Number image noisy before reseting: ",len(self.dataset.noise_index))
                self.reset_train_ds(update_probe_now,update_clean_now,lr)


        # checkpoint
        if self.time_for_evaluation(iteration, lr):
          logging.info(message)
          self.evaluate(iteration, lr)
          self.save_model(iteration)
          self.summary_writer.flush()
          if FLAGS.active :
              
              non_clean_index = list(set([i for i in range(len(self.dataset.all_data_label))]).difference(set(self.dataset.clean_index)))
              print("Average count of clean: ",np.mean(self.dataset.all_predict_logit[1][self.dataset.clean_index]))
              print("Average count of non-clean: ",np.mean(self.dataset.all_predict_logit[1][non_clean_index]))
              logging.info(str(("Previous adding acc: ",self.dataset.all_adding_acc)))
              logging.info(str(("List update probe: ",list_update_probe)))
              logging.info(str(("List update clean: ",list_update_clean)))
              logging.info(str(("List acc probe: ",self.dataset.all_probe_acc)))
          
        if FLAGS.gcloud and iteration == 20000:
          upload_checkpoint(FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,FLAGS.checkpoint_path)
        if iteration == 86000:
          break
      # end of iterations
      pbar.close()
      if FLAGS.gcloud:
        upload_checkpoint(FLAGS.dst_bucket_project,FLAGS.dst_bucket_name,FLAGS.checkpoint_path)
