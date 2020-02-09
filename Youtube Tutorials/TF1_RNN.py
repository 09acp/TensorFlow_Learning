3# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:57:18 2020

@author: 09ale
"""

""" 
RNN REGRESSION WITH TF 1 
[Youtube](https://www.youtube.com/watch?v=_NMI8peAmNA)
"""

import numpy as np 
#import tensorflow as tf
" TO USE TF 1 "
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
#%%
" Data for regression "

# read data
# data.index = pd.to_datetime(data.index) # convert index to time-series
# data.plot()
# data.info()
# split train / test
# scale fit&tranform train with MinMaxScaler 0-1
# transform test 

#%%

""" 
Applying batch  function 
- reading data in batches
"""
def next_batch(training_data, batch_size, steps):
    #random starting point
    rand_start = np.random.randint(0, len(training_data)-steps)
    # create labels for batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)

#%%
" RNN Parameters "
n_inputs = 1
n_steps = 12
n_neurons = 100
n_outputs = 1
learning_rate = 0.01
n_epochs = 4000
batch_size = 10 

#%%
" Placeholders X,Y "
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

" Loss Functions & Optimizer "
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate )
train = optimizer.minimize(loss)

" Initialize global variables "
init = tf.global_variables_initializer()

" Create an instance of tf.train.Saver() "
saver = tf.train.Saver()

#%%
" Create session and run it "
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs): 
        # fetches data 
        X_batch, y_batch = next_batch(train_X, batch_size, n_steps)
        # train data, label data 
        sess.run( train, feed_dict= {X:X_batch, y:y_batch})
        if epoch % 100 == 0:
            mse = loss.eval(feed_dict {X,X_batch, y:y_batch})
            print(epoch,"\tMSE", mse)
    saver.save(sess, "./TF1_RNN_model") #save model

#%%
" Predict test-set with model for 12 test samples "
with tf.Session() as sess:
    # Saver instance restores the rnn model
    saver.restore(sess, "./TF1_RNN_model")
    train_seed = list(train_X[12:]) # get predicted 
    
    for epoch in range(12): # 12 test samples
        X_batch = np.array(train_seed[-n_steps]).reshape(1, n_steps,1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0,-1,0])
     
#%%
" Reshape results "
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

df_test["Forecast"] = results
df_test.plot()


# =============================================================================
#                          END 
# =============================================================================





