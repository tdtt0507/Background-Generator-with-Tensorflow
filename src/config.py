
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
from prepro import prepro
train_dir = os.path.join('..','data','image_data','train_data') # Train dataset directory
dev_dir   = os.path.join('..','data','image_data','dev_data')  # validation dataset directory
test_dir  = os.path.join('..','data','image_data','test_data')

processed_dir = os.path.join('..','data','processed_data','processed_default')      # processed data 

processed_train_dir = os.path.join(processed_dir, )
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


# In[2]:


train_record_file = os.path.join(processed_dir,'train.tfrecord')
dev_record_file = os.path.join(processed_dir,'dev.tfrecord')
test_record_file = os.path.join(processed_dir,'test.tfrecord')


# In[3]:


flags = tf.flags

# # ============================================
flags.DEFINE_string("mode","train","train / prepro / test")
# # ============================================

flags.DEFINE_string("train_dir", train_dir,"") # Training file directory
flags.DEFINE_string("dev_dir", dev_dir,"") # Validation datset directory
flags.DEFINE_string("test_dir", test_dir,"") # Test dataset directory
flags.DEFINE_string("processed_dir",processed_dir,"") # processed dataset directory


# In[4]:


flags.DEFINE_string("target_dir",target_dir,"")
flags.DEFINE_string("model_ckpt_dir",model_ckpt_dir,"")
flags.DEFINE_string("train_log_dir",train_log_dir,"")
flags.DEFINE_string("result_dir",result_dir,"")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")

flags.DEFINE_integer("capacity", 15000, "Loaded datset to ram")
flags.DEFINE_integer("num_threads", 4, "Number of thread for input pipeline")
flags.DEFINE_boolean("use_cudnn",False, "Wheater to use GPU or CPU")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("epoch", 40, "Number of Epoch")
flags.DEFINE_integer("check_point",1000,"Checkpoint per step")
flags.DEFINE_float("init_lr", 0.5, "Initial Learning rate for adadelta")
flags.DEFINE_float("keep_prob", 0.7,"Dropout rate")
flags.DEFINE_integer("patience", 3, "Patience for lr decay")


# In[4]:


def main():
    config = flags.FLAGS
    if config.mode == 'prepro':
        print('In prepro')
        prepro(config)
    elif config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    else:
        print('Unknown mode')
        print('Select , prepro / train / test ')
        exit(0)
        
if __name__ == '__main__':
    main()

