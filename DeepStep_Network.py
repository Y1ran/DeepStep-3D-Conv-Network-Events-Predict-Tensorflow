# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:47:08 2018

@author: Administrator
"""

#import pyparams
import tensorflow as tf
import numpy as np
import math

'''M_IN_K = 1000.0,RANGE:1000000,S_LEN:12 = 32'''
'''M_IN_K = 1000.0,RANGE:1000,S_LEN:12 = 32'''
'''M_IN_K = 1000.0,RANGE:10,S_LEN:12 = 33.47'''
'''M_IN_K = 1000.0,RANGE:10,S_LEN:120 = 32'''
'''M_IN_K = 1000.0,RANGE:10,S_LEN:1200 = 33'''
'''M_IN_K = 1000.0,RANGE:1000,S_LEN:1200 = 33'''

#video trace path setting,
LogFile_Path = "./log/"   
NN_MODEL = "./submit/results/" # model path settings
class StepNetwork:
    def __init__(self, args, sess, idx=0, name=None):
     #  params = get_para.get_params()
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.frame_size = args.frame_size
        self.channels = args.channels

        self.num_gpu = args.num_gpu
        self.sess = sess
        self.batch_size = args.batch_size
    
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.frame_size, \
                                                  self.img_h, self.img_w, self.channels],
                                    name="input_videos")
        self.labels = tf.placeholder(tf.float32, [self.batch_size,1, self.img_h,\
                                                self.img_w, self.channels], name='labels')
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.dropout = tf.cond(self.is_train, lambda: args.dropout, lambda: 0.0)
        self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), 
                                           trainable=False, name='global_step')
        
     # Initail session or something
        with tf.variable_scope('Step-Conv'):
            x = tf.nn.relu(self.conv3d(self.images, 3, 1, 1, 2, 128, "conv1"))
            x = tf.nn.relu(self.conv3d(x, 1, 3, 3, 128, 128, "conv2"))
            x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 128, 256, "conv3"))
            x = tf.nn.relu(self.conv3d(x, 5, 1, 1, 256, 128, "conv4"))
            x = tf.nn.relu(self.conv3d(x, 1, 3, 3, 128, 128, "conv5"))
            x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 128,  64, "conv6"))
            x = tf.nn.relu(self.conv3d(x, 7, 1, 1, 64,    2, "conv7"))

            '''x=(1, 64, 64, 2)
          # x = tf.transpose(x, perm=[0,1,4,2,3])'''
            '''dropout layer这个后面再加'''
          
            logits = x
            '''logits是一个按像素和频道加起来的整数值？'''
        with tf.name_scope("loss"):
            '''I add Weight-decay here'''
            output = tf.floor(logits + 0.5)
            rmse_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels,\
                                                                output))))
            tf.summary.scalar("loss" + '_RMSE', rmse_loss)
            weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses', scope="model_%d" % idx))
            tf.summary.scalar("loss" + '_weight_decay_loss', weight_decay_loss)
            self.total_loss = rmse_loss + weight_decay_loss
            tf.summary.scalar("loss" + '_total_loss', self.total_loss)
            
        with tf.name_scope("inference"):
            output = lambda f_out:math.floor(x + 0.5)(logits)
            # add 2 channel for leaving and arrive
#            print(np.shape(output))
            self.infer_op = output

        self.tvars = tf.trainable_variables()
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
            
    def variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var
    
    def output(logits):
        output = tf.floor(logits + 0.5)
        return output
                
    def conv3d(self, x, fs, hs, ws, n_in, n_out, name=None):
        w = self.variable_with_weight_decay(shape=[fs, hs, ws, n_in, n_out], name=name+"_w", wd=0.0005)
        b = self.variable_with_weight_decay(shape=[n_out], name=name+"_bias")
        x = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name)
        return tf.nn.bias_add(x, b)
    
    def linear(self, x, units, name=None):
        w = self.variable_with_weight_decay(shape=[x.get_shape().as_list()[-1], units], name=name+"_w", wd=0.0005)
        b = self.variable_with_weight_decay(shape=[units], name=name+"_bias")
        return tf.nn.bias_add(tf.matmul(x, w), b)
    
    def build_feed_dict(self, images, labels, is_train):
        return {self.images: images, self.labels: labels, self.is_train: is_train}

    def variable_with_weight_decay(self, name, shape, wd=None):
        var = self.variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
        if wd is not None:
          weight_decay = tf.nn.l2_loss(var) * wd
          tf.add_to_collection('weightdecay_losses', weight_decay)
        return var


    def get_params(self):
        your_params = []
        return your_params

def restore_models(args, sess, models):
    new_models = []
    for model in models:
        restore_dict = restore_func(args.load_path, model.tvars)
        model.saver = tf.train.Saver(restore_dict)
        model.saver.restore(sess, args.load_path)
        new_models.append(model)
    return new_models

def restore_func(load_path, tvars):
    reader = tf.train.NewCheckpointReader(load_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_name = []
    for key in var_to_shape_map:
        var_name.append(key)
        var_name_sorted = sorted(var_name)

    convert_name = []
    for name in var_name_sorted:
        if "var_name" in name:
            name = name.replace("var_name", "StepNetwork")
        if "bc" in name:
            name = name.replace("bc", "Step-Conv")
            name += "_bias"
        elif "bout" in name:
            name = name.replace("bout", "logits_bias")
            pass
        elif "wc" in name:
            name = name.replace("wc", "Step-Conv")
            name += "_w"
        elif "wout" in name:
            name = name.replace("wout", "logits_w")
        else:
            pass
        convert_name.append(name)

    convert_dict = dict(zip(convert_name, var_name_sorted))
    restore_dict = dict()
    for v in tvars:
        tensor_name = v.name.split(':')[0]
        if reader.has_tensor(convert_dict[tensor_name]):
      # print('has tensor ', tensor_name)
          restore_dict[convert_dict[tensor_name]] = v
    restore_dict.pop("var_name/wout")
    restore_dict.pop("var_name/bout")
    return restore_dict

def get_multi_gpu_models(args, sess):
    models = []
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_idx in range(args.num_gpu):
            with tf.name_scope("model_%d" % gpu_idx), tf.device("/gpu:%d"%gpu_idx):
                Step = StepNetwork(args, sess, gpu_idx, name="StepNetwork")
                tf.get_variable_scope().reuse_variables()
                models.append(Step)
    return models


class MultiGPU(object):
    def __init__(self, args, models, sess):
        self.model = models[0]
        self.learning_rate = args.learning_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = self.model.global_step
        self.summary = self.model.summary
        self.models = models
        self.max_grad_norm = 100
        self.sess = sess
        
        loss_list = []
        infer_list = []
        grads_list = []


        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idx, model in enumerate(self.models):
                with tf.name_scope("grads_%d" % gpu_idx), tf.device("/gpu:%d" % gpu_idx):
                    loss = model.total_loss
                    loss_list.append(loss)
                    grads_and_vars = self.optimizer.compute_gradients(loss)
                    grads_and_vars = [(g, v) for g, v in grads_and_vars]
                    grads_list.append(grads_and_vars)
                    infer_list.append(model.infer_op)
                    tf.get_variable_scope().reuse_variables()
                    
        self.grads_and_vars = self.average_gradients(grads_list)
        self.loss = tf.add_n(loss_list) / len(loss_list)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        self.infer_list = tf.concat(infer_list, axis=0)
        self.grads_and_vars = self.average_gradients(grads_list)


    def train(self, sess, images, labels):
        half_idx = len(images) // 2
        feed_dict = {}
        for idx, model in enumerate(self.models):
            images_feed = images[idx* half_idx: (idx+1)*half_idx]
            labels_feed = labels[idx* half_idx: (idx+1)*half_idx]
            feed_dict.update(model.build_feed_dict(images_feed, labels_feed, True))
        return sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)

    def test(self, sess, images, labels):
      
        half_idx = len(images) // 2
        feed_dict = {}
      
        for idx, model in enumerate(self.models):
            images_feed = images[idx * half_idx: (idx + 1) * half_idx]
            labels_feed = labels[idx * half_idx: (idx + 1) * half_idx]
            feed_dict.update(model.build_feed_dict(images_feed, labels_feed, False))
        return sess.run(self.infer_list, feed_dict=feed_dict)
  
    def average_gradients(self, grads_list):
        average_grads = []
        for grad_and_vars in zip(*grads_list):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = np.expand_dims(g,axis=0)
                grads.append(expanded_g)
                
            grad = np.concatenate(grads, axis=0)
            grad = np.mean(grad, axis=0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
            
        return average_grads


class SingleGPU(object):
    def __init__(self, args, models):
        self.model = models[0]
        self.max_grad_norm = args.max_grad_norm
        self.learning_rate = args.learning_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = self.model.global_step
        self.summary = self.model.summary
        self.models = models

        loss_list = []
        grads_list = []
        infer_list = []

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idx, model in enumerate(self.models):
                with tf.name_scope("grads_%d"%gpu_idx), tf.device("/gpu:%d"%gpu_idx):
                    loss = model.total_loss
                    grads_and_vars = self.optimizer.compute_gradients(loss)
                    grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
                    loss_list.append(loss)
                    grads_list.append(grads_and_vars)
                    infer_list.append(model.infer_op)
                    tf.get_variable_scope().reuse_variables()

        self.loss = tf.add_n(loss_list) / len(loss_list)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        self.infer_list = tf.concat(infer_list, axis=0)
        
        return self.loss

        ''' def train(self, sess, images, labels):
           half_idx = len(images) // 2
           feed_dict = {}
           for idx, model in enumerate(self.models):
               images_feed = images[idx* half_idx: (idx+1)*half_idx]
               labels_feed = labels[idx* half_idx: (idx+1)*half_idx]
               feed_dict.update(model.build_feed_dict(images_feed, labels_feed, True))
               return sess.run([self.train_op, self.loss, self.accuracy, self.summary], feed_dict=feed_dict)
         '''
    def test(self, sess, images, labels):
        
        half_idx = len(images) // 2
        feed_dict = {}
        
        for idx, model in enumerate(self.models):
            images_feed = images[idx* half_idx: (idx+1)*half_idx]
            labels_feed = labels[idx* half_idx: (idx+1)*half_idx]
            feed_dict.update(model.build_feed_dict(images_feed, labels_feed, False))
            
        return sess.run(self.infer_list, feed_dict=feed_dict)

