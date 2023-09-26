import tensorflow as tf
import numpy as np
import csv
import random

# content-based filtering with neural network
def Main():
    # Build the User Model and Item Model
    User_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    Item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    # Create the User input and point to the base network
    User_Input = tf.keras.layers.Input(shape=())
if __name__ == '__main__':
    Main()