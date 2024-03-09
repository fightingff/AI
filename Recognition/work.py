import tensorflow as tf
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
import tensorflow.keras as Net
import matplotlib.pyplot as plt
import random

path = "./Images/"
train_datas,train_labels = [],[]
test_datas,test_labels = [],[]

def Shuffle(datas,labels):
    N = len(datas)
    index = [i for i in range(N)]
    random.shuffle(index)
    datas = datas[index]
    labels = labels[index]
    return datas,labels

def Load(N_Set=20,Size=3000):
    global train_datas,train_labels,test_datas,test_labels
    for i in range(N_Set):
        print("Loading "+str(i)+"...")
        label = tf.one_hot(i,20)
        for j in range(Size):
            target = cv.imread(path+str(i)+"/"+str(j)+".png")
            target = cv.cvtColor(target,cv.COLOR_BGR2GRAY)
            target = cv.resize(target,(28,28)).reshape((28,28))
            train_datas.append(target)
            train_labels.append(label)
    datas,labels = shuffle(train_datas,train_labels)
    N = len(train_datas)
    train_datas = np.array(datas[N//10:])
    train_datas = np.expand_dims(train_datas,axis=3)
    train_labels = np.array(labels[N//10:])
    test_datas = np.array(datas[:N//10])
    test_datas = np.expand_dims(test_datas,axis=3)
    test_labels = np.array(labels[:N//10])
    
def Train():
    Model = Net.Sequential([
        Net.layers.Convolution2D(input_shape=(28, 28, 1), filters=1, kernel_size=3, strides=1, padding='same', activation='relu'),
        Net.layers.MaxPooling2D(pool_size=2, strides=2),
        Net.layers.Convolution2D(filters=1,kernel_size=3, strides=1, padding='same', activation='relu'),
        Net.layers.MaxPooling2D(2, 2),
        Net.layers.Flatten(),
        Net.layers.Dense(128, activation='relu'),
        Net.layers.Dense(64, activation='sigmoid'),
        Net.layers.Dense(64, activation='sigmoid'),
        Net.layers.Dense(20, activation='softmax')
    ])
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    Records = Model.fit(train_datas, train_labels, batch_size=32, epochs=1000, validation_data=(test_datas, test_labels))
    Model.evaluate(test_datas, test_labels)
    Model.save('model.h5')
    
    plt.plot(Records.history['loss'])
    plt.show()
    
if (__name__ == '__main__'):
    Load(N_Set=20,Size=3000)
    Train()
    
