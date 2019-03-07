# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:47:08 2018

@author: Administrator
"""

import numpy as np
import utils
import time
import os

from DeepStep import get_multi_gpu_models, MultiGPU,restore_models
import tensorflow as tf

def valid(args, test_file, trainer, sess):
    next_start_pos = 0
    infer_list = []
    labels = []
    num_test_videos =  len(list(open(test_file)))
    all_steps = int((num_test_videos - 1) / (args.batch_size * args.num_gpu) + 1)
    for step in range(all_steps):
        start_time = time.time()
        test_images, test_labels, next_start_pos, _, valid_len = utils.read_clip_and_label(
                    test_file, args.batch_size * args.num_gpu, start_pos=next_start_pos)
        infer_list.extend(trainer.test(sess, test_images, test_labels))
        labels.extend(test_labels)

    return np.mean(np.equal(infer_list[:num_test_videos], labels[:num_test_videos])), \
                                                time.time()-start_time

def main(args):
    save_dir = os.path.join(args.save_dir)
    log_dir = os.path.join(args.log_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    summary_writer = tf.summary.FileWriter(log_dir)
    config_proto = utils.get_config_proto()

    sess = tf.Session(config=config_proto)
    models = get_multi_gpu_models(args, sess)
#    trainer = MultiGPU(args, models, sess)
    model = models[0]
    saver=tf.train.Saver(max_to_keep=1)
    train_loss = model.total_loss
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(train_loss)
    sess.run(tf.global_variables_initializer())
    
    if args.use_pretrained:
        models = restore_models(args, sess, models)
        model = models[0]

    for step in range(1, args.train_steps + 1):
        step_start_time = time.time()
        train_images, train_labels = utils.read_to_batch(sess, filename='3D-Block-2016-05.npy',
          batch_size=args.batch_size * args.num_gpu)
        images_batch,labels_batch = sess.run([train_images,train_labels])
#        print(type(train_images),type(train_labels))
        feed_dict = model.build_feed_dict(images_batch, labels_batch, True)
        _,loss_value = sess.run([train_step, train_loss],feed_dict=feed_dict)
#        _, loss,summaries = trainer.train(sess, images_batch, labels_batch)
#        _, loss, accuracy, summaries = trainer.train(sess, train_images, train_labels)
        summaries = sess.run(model.summary,feed_dict=feed_dict)
        summary_writer.add_summary(summaries, step)

        if step % args.log_step == 0:
            print ("step %d, loss %.5f, time %.2fs" % (step, loss_value, time.time() - step_start_time))

#        if step % args.eval_step == 0:
#            val_accuracy, test_time = valid(args, 'list/testlist.txt', trainer, sess)
#            print ("test accuracy: %.5f, test time: %.5f" % (val_accuracy, test_time))
    saver.save(sess,'./save/StepNetwork.ckpt', global_step=args.train_steps)

class args(object):
    
    def __init__(self, heigh, width, frame_size, channels, gpu_num, batch_size, \
                 dropout, use_pretrained, train_steps,learning_rate):
        self.img_h = heigh
        self.img_w = width
        self.frame_size = frame_size
        self.channels = channels
        self.num_gpu = gpu_num
        self.batch_size = batch_size
        self.save_dir = './save/'
        self.log_dir = './log/'
        self.dropout = dropout
        self.use_pretrained = use_pretrained
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.log_step = 1
        
        
if __name__ == '__main__':
    args = args(64, 64, 7, 2, 2, 10, dropout=0.5, use_pretrained=False,\
                train_steps=100,learning_rate=1e-3)

    main(args)