#!/usr/bin/env python3
import matplotlib.pyplot as plt
import lifereader
import numpy as np
import tensorflow as tf
import math
import timeit

board = lifereader.readlife('lifep/BREEDER3.LIF', 2048)

plt.figure(figsize=(20,20))
plotstart=924
plotend=1124
plt.imshow(board[plotstart:plotend,plotstart:plotend])

plt.figure(figsize=(20,20))
plt.imshow(board)
plt.show()

#tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)

boardtf = tf.cast(board, dtype=tf.float16)

@tf.function
def runlife(board, iters):
    # Init work
    
    for i in range(iters):
        # In each iteration, compute two bool tensors
        # ’survive’ and ’born’: TODO
        
        # Then, update the board by keeping these tensors
        board = tf.cast(tf.logical_or(survive, born), board.dtype)
        
    # Final work
    return board


# tic = timeit.default_timer()
# boardresult = runlife(boardtf, 1000);
# toc = timeit.default_timer();
# print("Compute time: " + str(toc - tic))
# result = np.cast[np.int32](boardresult);
# print("Cells alive at start: " + str(np.count_nonzero(board)))
# print("Cells alive at end:   " + str(np.count_nonzero(result)))
# print(np.count_nonzero(result))
# plt.figure(figsize=(20,20))
# plt.imshow(result[plotstart:plotend,plotstart:plotend])
