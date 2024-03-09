import matplotlib.pyplot as plt
import numpy as np
# ordinary least squares
with open("data.txt", 'r') as f:
    data = f.read().split()
    N = int(data[0])
    D = int(data[1])
    X, Y = [], []
    for i in range(2, len(data),2):
        X.append(float(data[i]))
        Y.append(float(data[i+1]))
    X_ = sum(X) / N
    Y_ = sum(Y) / N
    X__ = sum([x*x for x in X])/N
    XY_ = sum([x*y for x,y in zip(X,Y)])/N
    W = (XY_ - X_*Y_)/(X__ - X_*X_)
    b = Y_ - W*X_
    print (W, b)
    
    
    N = int(N) // 2
    for i in range(N):
        print((Y[i+N] - Y[i]) / (Y[i] * X[i+N] - Y[i+N] * X[i]))
    
    plt.figure("Rt - t关系图")
    plt.xlabel("Rt /Ω")
    plt.ylabel("t /℃")
    plt.scatter(X,Y)
    plt.plot(X,[W*x+b for x in X],color='r')
    plt.xticks(np.arange(26, 72, 5))
    plt.yticks(np.arange(55, 66, 2))
    plt.text(27,64,"Rt = %.6f * t + %.6f"%(W,b),fontsize=15)
    print(W/b)
    plt.title("Rt - t Relation",size=20)
    for x,y in zip(X,Y):
        plt.text(x+1.5,y,"(%.1f, %.2f)"%(x,y),fontsize=8)
    # with open("data.out","r") as pp:
    #     data = pp.read().split()
    #     W = float(data[1])
    #     b = float(data[0])
    #     plt.plot(X,[W*x+b for x in X],color='g')
    plt.show()    