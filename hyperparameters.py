# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""


import os
pwd = os.path.dirname(os.path.abspath(__file__))
from classifier_multi_label.utils import load_vocabulary,load_third_fourth_dict


class Hyperparamters:
    # Train parameters
    print_step = 10 
    batch_size = 64           
    summary_step = 10
    num_saved_per_epoch = 3
    max_to_keep = 100
    logdir = 'logdir/model_02'
    
    # Load vocabulcary dict
    dict_id2label,dict_label2id = load_vocabulary(os.path.join(pwd,'data','vocabulary_label.txt'))
    label_vocabulary = list(dict_id2label.values())
    third_fourth_pos_dict,fourth_third_neg_dict = load_third_fourth_dict(os.path.join(pwd,'data','notebook_fourth_label.xlsx'))
    
    # Optimization parameters
    num_train_epochs = 20
    warmup_proportion = 0.1    
    use_tpu = None
    do_lower_case = True    
    learning_rate = 5e-5     
    
    # TextCNN parameters
    num_filters = 128    
    filter_sizes = [2,3,4,5,6,7]
    embedding_size = 384
    keep_prob = 0.5
    
    # Sequence and Label
    sequence_length = 60
    num_labels = len(list(dict_id2label))    
 
    # ALBERT
    model = 'albert_small_zh_google'
    bert_path = os.path.join(pwd,model)
    data_dir = os.path.join(pwd,'data')
    vocab_file = os.path.join(pwd,model,'vocab_chinese.txt')
    init_checkpoint = os.path.join(pwd,model,'albert_model.ckpt')
    saved_model_path = os.path.join(pwd,'model')    
    
   
    
    
    

if __name__ == '__main__': 
    #
    hp = Hyperparamters()
    print(hp.third_fourth_pos_dict)


    
    