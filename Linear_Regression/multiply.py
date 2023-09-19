import matplotlib.pyplot as plt
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
    
    plt.figure()
    plt.scatter(X,Y)
    plt.plot(X,[W*x+b for x in X],color='r')
    with open("data.out","r") as pp:
        data = pp.read().split()
        W = float(data[1])
        b = float(data[0])
        plt.plot(X,[W*x+b for x in X],color='g')
    plt.show()    