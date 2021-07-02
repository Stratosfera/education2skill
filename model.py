# -*- coding: utf-8 -*-

import keras
from keras.layers import Dense, Input#, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from focal_loss import BinaryFocalLoss
import tensorflow as tf


def get_model(X_train, y_train):

    #MODEL
    inputs = Input(shape=(X_train[0].shape[0],))
    x = Dense(1000, activation='relu')(inputs)
    #x= Dropout(0.1)(x)
    x = Dense(1000, activation='relu')(x)
    #x= Dropout(0.1)(x)
    
    
    # level 1
    x_level1 = Dense(1000, activation='relu')(x)
    #x_level1 = Dropout(0.1)(x_level1)
    preds_level1 = Dense(y_train[0,1].toarray().shape[1], activation='sigmoid', name="preds_level1")(x_level1)
    
    # level 2
    x_level2 = concatenate([x, x_level1])
    x_level2 = Dense(1000, activation='relu')(x_level2)
    #x_level2 = Dropout(0.1)(x_level2)
    preds_level2 = Dense(y_train[0,2].toarray().shape[1], activation='sigmoid', name="preds_level2")(x_level2)
    
    # level 3
    x_level3 = concatenate([x, x_level1, x_level2])
    x_level3 = Dense(1000, activation='relu')(x_level3)
    #x_level3 = Dropout(0.1)(x_level3)
    preds_level3 = Dense(y_train[0,3].toarray().shape[1], activation='sigmoid', name="preds_level3")(x_level3)
    
    # level 4
    x_level4 = concatenate([x, x_level1, x_level2, x_level3])
    x_level4 = Dense(1000, activation='relu')(x_level4)
    #x_level4 = Dropout(0.1)(x_level4)
    preds_level4 = Dense(y_train[0,0].toarray().shape[1], activation='sigmoid', name="preds_level4")(x_level4)
    
    model = Model(inputs=[inputs], outputs=[preds_level1, preds_level2, preds_level3, preds_level4])
    
    # compile the keras model
    model.compile(loss=BinaryFocalLoss(gamma=3), 
                  optimizer='adam',
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold = 0.4),
                           keras.metrics.Precision(thresholds=0.4),
                           keras.metrics.Recall(thresholds=0.4)])
    
    model.summary()
    
    return model





