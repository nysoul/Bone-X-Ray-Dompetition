from __future__ import print_function

import os
import time
import datetime
import random
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.utils import np_utils

import data_loader

def run_MURA(batch_size,nb_epoch,depth,nb_dense_block,nb_filter,growth_rate,dropout_rate,learning_rate,weight_decay):
    
    ###################
    # Data processing #
    ###################

    
    im_size = 320            
    X_train_path, Y_train = data_loader.load_path(root_path = 'train',size = im_size)  
    X_valid_path, Y_valid = data_loader.load_path(root_path = 'valid', size = im_size)  

    X_valid = data_loader.load_image(X_valid_path,im_size)  
    Y_valid = np.asarray(Y_valid)
    nb_classes = 1                                
    img_dim = (im_size,im_size,1)                 

    
    ###################
    # Construct model #
    ###################

    model = densenet.DenseNet(nb_classes,img_dim,depth,nb_dense_block,growth_rate,nb_filter,dropout_rate=dropout_rate,weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])

    
    ####################
    # Network training #
    ####################

    print("Start Training")

    list_train_loss = []
    list_valid_loss = []
    list_learning_rate = []
    best_record = [100,0,100,100]
    start_time = datetime.datetime.now()
    for e in range(nb_epoch):

        if e == int(0.25 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 50.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = len(X_train_path) / split_size
        arr_all = np.arange(len(X_train_path)).astype(int)
        random.shuffle(arr_all)                 
        arr_splits = np.array_split(arr_all, num_splits)

        l_train_loss = []
        batch_train_loss = []
        start = datetime.datetime.now()

        for i,batch_idx in enumerate(arr_splits):


            X_batch_path,Y_batch = [],[]
            for idx in batch_idx:
                X_batch_path.append(X_train_path[idx])
                Y_batch.append(Y_train[idx])
            X_batch = data_loader.load_image(Path = X_batch_path, size =im_size)
            Y_batch = np.asarray(Y_batch)
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])
            batch_train_loss.append([train_logloss, train_acc])
            if i %100 == 0:
                loss_1, acc_1 = np.mean(np.array(l_train_loss), 0)
                loss_2, acc_2 = np.mean(np.array(batch_train_loss), 0)
                batch_train_loss = []           
                print ('[Epoch {}/{}] [Batch {}/{}] [Time: {}] [all_batchs--> train_epoch_logloss: {:.5f}, train_epoch_acc:{:.5f}] '.format
                    (e+1,nb_epoch,i, len(arr_splits),datetime.datetime.now() - start,loss_1,acc_1),
                    '[this_100_batchs-->train_batchs_logloss: {:.5f}, train_batchs_acc:{:.5f}]'.format(loss_2, acc_2))

        
        valid_logloss, valid_acc = model.evaluate(X_valid,Y_valid,verbose=0,batch_size=64)

        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_valid_loss.append([valid_logloss, valid_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))

        # to convert numpy array to json serializable
        print('[Epoch %s/%s] [Time: %s, Total_time: %s]' % (e + 1, nb_epoch, datetime.datetime.now() - start,
            datetime.datetime.now() - start_time),end = '')
        print('[train_loss_and_acc:{:.5f} {:.5f}] [valid_loss_acc:{:.5f} {:.5f}]'.format(list_train_loss[-1][0],
            list_train_loss[-1][1],list_valid_loss[-1][0],list_valid_loss[-1][1]))


        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["valid_loss"] = list_valid_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log/experiment_log_MURA.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)

        record = [valid_logloss,valid_acc,abs(valid_logloss-list_train_loss[-1][0]),abs(valid_acc-list_train_loss[-1][1]),]
        if ((record[0]<=best_record[0]) &(record[1]>=best_record[1])) :
            if e <= int(0.25 * nb_epoch)|(record[2]<=best_record[2])&(record[3]<=best_record[3]):
                best_record=record                      
                print('saving the best model:epoch',e+1,best_record)
                model.save('save_models/best_MURA_modle@epochs{}.h5'.format(e+1))
        model.save('save_models/MURA_modle@epochs{}.h5'.format(e+1))


if __name__ == '__main__':

    list_dir = ["./log", "./figures", "./save_models"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)
    batch_size=4
    nb_epoch=3
    depth=22
    nb_dense_block=4
    nb_filter=16
    growth_rate=12
    dropout_rate=0.2
    learning_rate=1E-3
    weight_decay=1E-4

    run_MURA(batch_size,nb_epoch,depth,nb_dense_block,nb_filter,growth_rate,dropout_rate,learning_rate,weight_decay)
