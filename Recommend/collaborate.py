import tensorflow as tf
import numpy as np
import csv 
import random

# collaborative filtering
def Init_Movie(movie):
    cat = dict()
    X = np.zeros((1,1))
    with open('movies.csv',encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        for _i,x in enumerate(data):
            movie[x['movieId']] = _i
            for type in x['genres'].split('|'):
                if type not in cat:
                    cat[type] = len(cat)
        X = np.zeros((len(movie),len(cat)))
        for x in data:
            for type in x['genres'].split('|'):
                X[movie[x['movieId']],cat[type]] = 1
    f.close()
    return X

test_data = []
def Init_Data():
    global test_data
    with open('ratings.csv',encoding='utf-8') as f:
        train_data = list(csv.DictReader(f))
    f.close()
    random.shuffle(train_data)
    test_data = train_data[:len(train_data) // 5]
    return train_data[len(test_data):]

lamda = 1e-2
alpha = 1e-4
eps = 1e-5
# Cost function
def J(W,X,B,marks):
    K = W @ X.T + B
    S = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if marks[i,j] != -1:
                S += (K[i,j] - marks[i,j]) ** 2 
    S += lamda* np.sum(W ** 2) + lamda * np.sum(X ** 2)
    return S * 0.5
    
def Dw(W,X,B,marks,K):
    dW = np.zeros(W.shape)
    for i in range(marks.shape[0]):
        for j in range(marks.shape[1]):
            if marks[i,j] != -1:
                dW[i] += K[i,j] * X[j]
    dW += lamda * W
    return dW

def Db(W,X,B,marks,K):
    dB = np.zeros(B.shape)
    for i in range(marks.shape[0]):
        for j in range(marks.shape[1]):
            if marks[i,j] != -1:
                dB[i,j] += K[i,j] 
    return dB

def Dx(W,X,B,marks,K):
    dX = np.zeros(X.shape)
    for i in range(marks.shape[0]):
        for j in range(marks.shape[1]):
            if marks[i,j] != -1:
                dX[j] += K[i,j] * W[i]
    dX += lamda * X
    return dX
    
def Main():
    data = Init_Data()
    user = dict()
    movie = dict()
    N,M,D = 0,0,0
    for x in data:
        if x['userId'] not in user:
            user[x['userId']] = N
            N += 1
        if x['movieId'] not in movie:
            movie[x['movieId']] = M
            M += 1

    X = Init_Movie(movie)
    D = X.shape[1]
    M = len(movie)
    marks = np.zeros((N,M)) - 1
    W = np.zeros((N,D))
    B = np.zeros((N,M))
    
    for x in data:
        marks[user[x['userId']],movie[x['movieId']]] = float(x['rating'])

    print(W.shape,X.shape,B.shape,marks.shape)
    cnt = 0
    global test_data
    while(J(W,X,B,marks) / (N * M) >eps and cnt < 100000):
        K = W @ X.T + B - marks
        
        # for dubug
        # print(K)
        # print("Users:")
        # print(W)
        # print("Movies:")
        # print(X)
        # print("Marks:")
        # print(W @ X.T + B)
        
        W -= alpha * Dw(W,X,B,marks,K)
        B -= alpha * Db(W,X,B,marks,K)
        X -= alpha * Dx(W,X,B,marks,K)
        cnt += 1
        
        S = 0
        for x in test_data:
            if x['userId'] in user and x['movieId'] in movie:
                i = user[x['userId']]
                j = movie[x['movieId']]
                if abs(W[i] @ X[j] + B[i,j] - float(x['rating'])) < 0.5:
                    S += 1
        print("%d %.1f%%"%(cnt,S / len(test_data) * 100))
    
    with open("model.txt") as f:
        f.write(str(W))
        f.write(str(X))
        f.write(str(B))
    f.close() 
    
if __name__ == '__main__':
    Main()