# -*- coding: utf-8 -*-


import numpy as np


def full_generator(X, y1, y2, y3, y4, batch_size, number_of_batches, shuffle, infinite=True):
    #number_of_batches = X.shape[0]/batch_size
    
    sample_index = np.arange(X.shape[0])
    
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)

    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        
        tmp_batch_size = len(X[batch_index])
        X_batch = np.empty((tmp_batch_size, X[0].shape[0]))
        y_batch1 = np.empty((tmp_batch_size, y1[0].shape[1]))
        y_batch2 = np.empty((tmp_batch_size, y2[0].shape[1]))
        y_batch3 = np.empty((tmp_batch_size, y3[0].shape[1]))
        y_batch4 = np.empty((tmp_batch_size, y4[0].shape[1]))        
        counter += 1
        
        for b in range(tmp_batch_size):
            X_text  =  X[batch_index][b]
            y_sparse1 = y1[batch_index][b]
            y_sparse2 = y2[batch_index][b]
            y_sparse3 = y3[batch_index][b]
            y_sparse4 = y4[batch_index][b]

                        
            X_batch[b,] = [X_text][0]
            y_batch1[b,] = y_sparse1.toarray()
            y_batch2[b,] = y_sparse2.toarray()
            y_batch3[b,] = y_sparse3.toarray()
            y_batch4[b,] = y_sparse4.toarray()
            
        yield X_batch, [y_batch1, y_batch2, y_batch3, y_batch4]           
        
        if (counter == number_of_batches):
            if not infinite:
                break
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
            