# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:44:15 2018

@author: Administrator
"""

import os
import PIL.Image as Image
import tensorflow as tf
import random
import numpy as np

import time


def read_to_batch(sess, filename, batch_size):
    '''first use batch = 1'''
    precedent_frames = []
    label_frames = []
    block = np.load(filename)
    
    precedent = np.zeros((7, 64, 64, 2))
    label = np.zeros((1, 64, 64, 2))
    
    for i in range(337, 1488):
        label = block[i, :, :, :]   # label是当前选择的frame
        label = np.reshape(label, (1,64,64,2))
        label_frames.append(label)
        
        precedent[0:2, :, :, :] = block[i-2:i, :, :, :]   # 使用过去的对应时段作为预测的frame, 这是前1小时
        precedent[2:4, :, :, :] = block[i-48:i-46, :, :, :]   # 前一天
        precedent[4:7, :, :, :] = block[i-337:i-334, :, :, :]  # 前一周
        precedent_frames.append(precedent)
    
    _, train_images, train_labels = generate_batch(sess, precedent_frames, label_frames,batch_size)
        
    return train_images, train_labels

def generate_batch(sess, features, labels, batch_size):
    
    features_placeholder = tf.placeholder(tf.float32, np.shape(features))
    labels_placeholder = tf.placeholder(tf.float32, np.shape(labels))
    #dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
    
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.repeat(batch_size)
    batched_dataset = dataset.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,\
                                              labels_placeholder: labels})

    batch_xs, batch_ys = iterator.get_next()
    
    return iterator.initializer,batch_xs, batch_ys

    
    
    
def get_frames_data(filename, num_frames_per_clip=16):
    
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        if(len(filenames)<num_frames_per_clip):
            return [], s_index
        filenames = sorted(filenames)
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
            
    return ret_arr, s_index

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, \
                        crop_size=112, shuffle=False):
    lines = open(filename,'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
  # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    np_mean = np.load('crop_mean.npy')
  # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = list(range(len(lines)))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
    # Process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if(batch_index>=batch_size):
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        if not shuffle:
      # print("Loading a video clip from %s..." % dirname)
            pass
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    img_datas = []
    
    if(len(tmp_data)!=0):
        for j in range(len(tmp_data)):
            img = Image.fromarray(tmp_data[j].astype(np.uint8))
            
            if(img.width>img.height):
                scale = float(crop_size)/float(img.height)
                img = np.array(np.resize(np.array(img),(int(img.width * scale + 1), \
                                            crop_size))).astype(np.float32)
            else:
                scale = float(crop_size)/float(img.width)
                img = np.array(np.resize(np.array(img),(crop_size, \
                                         int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
        data.append(img_datas)
        label.append(int(tmp_label))
        batch_index = batch_index + 1
        read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

def calculate_ucf_101_mean(ucf_train_lst, num_frames=16, new_w_h_size=112):
  mean = np.zeros((num_frames, new_w_h_size, new_w_h_size, 3))
  count = 0

  with open(ucf_train_lst) as f:
    for line in f:
      vid_path = line.split()[0]
      start_pos = int(line.split()[1])

      stack_frames = []

      for i in range(start_pos, start_pos+num_frames):
        img = os.path.join(vid_path, "{:06}.jpg".format(i))

        stack_frames.append(img)

      stack_frames = np.array(stack_frames)
      mean += stack_frames
      count += 1
  mean/=float(count)
  print (mean)
  return mean

# mean_ucf101_16 = calculate_ucf_101_mean("trainlist01.txt")
# np.save("crop_mean_16.npy", mean_ucf101_16)

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  
  config_proto.gpu_options.allow_growth = True
  
  return config_proto
