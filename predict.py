# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
pwd = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from classifier_multi_label.networks import NetworkAlbert
from classifier_multi_label.classifier_utils import get_feature_test,id2label
from classifier_multi_label.hyperparameters import Hyperparamters as hp
          

class ModelAlbertTextCNN(object,):
    """
    加载 NetworkAlbert 神经网络模型
    """
    def __init__(self):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            out_dir = os.path.join(pwd, "model")
            with sess.as_default():
                albert =  NetworkAlbert(is_training=False)
                saver = tf.train.Saver()  
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(out_dir,'small-google-gelu-V1.0'))
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

MODEL = ModelAlbertTextCNN()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's label.
    """
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids:[feature[2]],
          }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]   
    return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]    
      

    
def get_label_multi(sentences):
    """
    Prediction of some sentence's labels.
    """
    features = [get_feature_test(str(sentence)) for sentence in sentences ]
    fd = {MODEL.albert.input_ids: [feature[0] for feature in features],
          MODEL.albert.input_masks: [feature[1] for feature in features],
          MODEL.albert.segment_ids:[feature[2] for feature in features]}    
    predictions = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd) 
    return [[id2label(l) for l in np.where(prediction==1)[0] if l!=0]  for prediction in predictions]   



if __name__ == '__main__':
    import time
    start = time.time()
    sent = '制热效果很差劲 开到一档就很暖和了'
    print(get_label(sent))
    end = time.time()
    print(end-start)#57ms
    

#    third_class = get_notebook_label(sent)
#    print('third_class:',third_class)
#    sa = 0
#    fourth_class = get_fourth_by_third_and_sa(third_class,sa) 
#    print('fourth_class:',fourth_class)



    
    
    
    
