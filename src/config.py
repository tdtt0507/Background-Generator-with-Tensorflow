import tensorflow as tf
import os
import argparse as _argparse
from PIL import Image
import numpy as np
import collections
import copy
import random
from tqdm import tqdm
from prepro import prepro
from main import train

train_dir = os.path.join('..','data','image_data','train_data') # Train dataset directory
dev_dir   = os.path.join('..','data','image_data','dev_data')  # validation dataset directory
test_dir  = os.path.join('..','data','image_data','test_data')
processed_dir = os.path.join('..','data','processed_data')      # processed data 

target = "yongho_BG_with_attention"
target_dir = os.path.join('..','data','research_data',target)
model_ckpt_dir = os.path.join('..','data','research_data',target,'model_ckpt')
train_log_dir = os.path.join('..','data','research_data',target,'train_log')
result_dir = os.path.join('..','data','research_data',target, 'result')

os.makedirs(processed_dir) if not os.path.exists(processed_dir) else 0
os.makedirs(target_dir) if not os.path.exists(target_dir) else 0
os.makedirs(model_ckpt_dir) if not os.path.exists(model_ckpt_dir) else 0
os.makedirs(train_log_dir) if not os.path.exists(train_log_dir) else 0
os.makedirs(result_dir) if not os.path.exists(result_dir) else 0

train_record_file = os.path.join(processed_dir,'train.tfrecord')
dev_record_file = os.path.join(processed_dir,'dev.tfrecord')
test_record_file = os.path.join(processed_dir,'test.tfrecord')

flags = tf.flags


flags.DEFINE_integer("max_size", 500*1000, "")

flags.DEFINE_string("train_dir", train_dir,"") # Training file directory
flags.DEFINE_string("dev_dir", dev_dir,"") # Validation datset directory
flags.DEFINE_string("test_dir", test_dir,"") # Test dataset directory

flags.DEFINE_string("mode", "prepro", "prepro / train / test")
flags.DEFINE_string("target_dir",target_dir,"")
flags.DEFINE_string("model_ckpt_dir",model_ckpt_dir,"")
flags.DEFINE_string("train_log_dir",train_log_dir,"")
flags.DEFINE_string("result_dir",result_dir,"")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")

flags.DEFINE_integer("capacity", 10, "Loaded datset to ram")
flags.DEFINE_integer("num_threads", 4, "Number of thread for input pipeline")
flags.DEFINE_boolean("use_cudnn",False, "Wheater to use GPU or CPU")
flags.DEFINE_integer("batch_size", 1, "batch_size")

flags.DEFINE_integer('num_steps', 10000, "Steps of iteration")
flags.DEFINE_integer("epoch", 40, "Number of Epoch")
flags.DEFINE_integer("check_point",1000,"Checkpoint per step")
flags.DEFINE_float("init_lr", 0.5, "Initial Learning rate for adadelta")
flags.DEFINE_float("keep_prob", 0.7,"Dropout rate")
flags.DEFINE_integer("patience", 3, "Patience for lr decay")

flags.DEFINE_integer('block_size', 30, "Size of Picture partitioning Block")

# -------------------- Model realted Flags --------------------

flags.DEFINE_integer('hidden_size', 150, "RNN hidden_size")
flags.DEFINE_integer('cut_size', 4, "Cutting size of conved data")
# -------------------------------------------------------------

def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Unknown mode")
        exit(0)

if __name__ == "__main__":
    tf.app.run()
