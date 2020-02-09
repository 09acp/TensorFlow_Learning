3# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:57:18 2020

@author: 09ale
"""

""" 
LINEAR REGRESSION WITH TF 1
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

train_X = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
train_Y = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(train_X, train_Y, "x")

" Values for gradient and intercept ==> values will change "
m = tf.Variable ( 0.39)
b = tf.Variable ( 0.2)

#%%
error = 0 
for x,y in zip(train_X, train_Y):
    y_hat = m*x + b # prediction 
    error += (y-y_hat)**2 # cost minimization
    
" Optimization Functions "
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)    
train = optimizer.minimize(error)
    
" Initialize Variables "
init = tf.global_variables_initializer()
#%%
" Create Session and Run graph "
with tf.Session() as sess:
    sess.run(init) # initialize Variables 
    epochs = 100
    for i in range(epochs): 
        sess.run(train) # Run model
        final_slope, final_intercept = sess.run([m,b]) # Results

#%%
" Evaluate Results "
# Random m / b values 
0.39 , 0.2 
# Predicted values 
1.06 , -0.27

test_X = np.linspace(-1,11,10)
test_plot = final_slope * test_X + final_intercept

plt.plot(test_X, test_plot, "r" )
plt.plot(train_X,train_Y,"x")

# =============================================================================
#                          END 
# =============================================================================





