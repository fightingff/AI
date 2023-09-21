import numpy as np
import matplotlib.pyplot as plt

# anomaly detection
Eps = 1e-1 # threshold
def Fp(x,mu,dt):
    return 1 / np.sqrt(2 * np.pi * dt) * np.exp(- (x - mu)**2 / (2 * dt))

# draw 3D graph
def Draw(Mu,Dt):
    plt.figure()
    x = np.linspace(Mu[0,0] - 3 * np.sqrt(Dt[0]), Mu[0,0] + 3 * np.sqrt(Dt[0]), 100)
    y = np.linspace(Mu[0,1] - 3 * np.sqrt(Dt[1]), Mu[0,1] + 3 * np.sqrt(Dt[1]), 100)
    X,Y = np.meshgrid(x,y)
    Z = Fp(X,Mu[0,0],Dt[0]) * Fp(Y,Mu[0,1],Dt[1])
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=plt.cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.show()

def Main():
    N = 0 # number of data points
    D = 0 # number of dimensions
    data = [] # data points
    Mu = 0 # mean
    Dt = [] # parameter for variance
    with open("data.txt", "r") as f:
        data = f.read().split()
        data = [float(i) for i in data]
        N,D = int(data[0]), int(data[1])
        data = np.array(data[2:])
        data = data.reshape((N, D))
    f.close()
    
    Mu = np.zeros((1,D))
    for x in data:
        Mu += x
    Mu /= N

    for i in range(D):
        sum = 0
        for x in data:
            sum += (x[i] - Mu[0,i])**2
        Dt.append(sum / (N - 1))
    
    Draw(Mu,Dt)
    
    while 1:
        x = input("Enter data point: ")
        x = x.split(" ")
        x = [float(i) for i in x]
        x = np.array(x)
        if len(x) != D:
            print("Invalid data point")
            continue
        
        p = 1
        for i in range(D):
            p *= Fp(x[i],Mu[0,i],Dt[i])

        if p < Eps:
            print("Anomaly detected")
        else:
            print("Passed")
    
    
if __name__ == "__main__":
    Main()