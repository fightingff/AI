import numpy as np
import matplotlib.pyplot as plt

# k-means clustering
Kt = 100   # max iteration
K = 3      # number of clusters
Eps = 1e-3 # threshold
delay = 0.5 # delay time

def J(centers,datas):
    # calculate cost function
    Cost = 0
    for v in datas:
        minDist = 1e100
        for c in centers:
            minDist = min(minDist,np.linalg.norm(v-c))
        Cost += minDist
    return Cost

class Painter:
    Xs,Ys,Cs = [],[],[]
    def __init__(self,datas):
        self.Xs = datas[:,0]
        self.Ys = datas[:,1]
        self.Cs = ['b' for i in range(len(datas))]
        plt.figure()
        
    def DrawPoints(self,X,Y,C):
        plt.clf()
        # draw circles
        for i in range(len(X)):
            circle = plt.Circle((X[i],Y[i]),radius=100,color='r',fill=False,linestyle='--')
            plt.gca().add_patch(circle)
        
        # draw points
        X = list(X)
        Y = list(Y)
        X.extend(self.Xs)
        Y.extend(self.Ys)
        C.extend(self.Cs)
        plt.scatter(X,Y,c=C)
        
        plt.draw()
        plt.pause(delay)
        plt.close()

def Main():
   # input data
   N = 0
   D = 0
   global K
   global delay
   datas = np.array([])
   with open('data.txt', 'r') as f:
       f = [float(x) for x in f.read().split()]
       N,D = int(f[0]),int(f[1])
       datas = np.array(f[2:])
       datas = datas.reshape(N,D)
    
   Pic = Painter(datas)

   Ans = 1e100
   id = [i for i in range(N)]
   centers = np.zeros((K,D))
   for i in range(Kt):
       # random K centers
       id = np.random.permutation(id)
       for j in range(K):
           centers[j] = datas[id[j]]
        
       # k-means clustering
       lstK = K
       while True:
           # assign each data to the nearest center
           
           # visualization
        #    X = centers[:,0]
        #    Y = centers[:,1]
        #    C = ['r' for i in range(K)]
        #    Pic.DrawPoints(X,Y,C)
           
           Means = np.zeros((K,D))
           cnt = np.zeros(K)
           for v in datas:
                minDist = 1e100
                kth = 0
                for _i,c in enumerate(centers):
                    if np.linalg.norm(v-c) < minDist:
                        minDist = np.linalg.norm(v-c)
                        kth = _i
                Means[kth] += v
                cnt[kth] += 1
            
            # update centers
           lst = J(centers,datas) / N
           k = 0
           tk = K
           while k < tk:
               if cnt[k] != 0:
                   centers[k] = Means[k] / cnt[k]
               else:
                   centers=np.delete(centers,k,0)
                   K -= 1
                   k -= 1
                   tk -= 1
               k += 1
           if lst - J(centers,datas) / N < Eps:
               break
       Cost = J(centers,datas) / N
       if Cost < Ans:
              Ans = Cost
              AnsCenter = centers
       
       K = lstK
   
   print(Ans)
   print(AnsCenter)
   X = AnsCenter[:,0]
   Y = AnsCenter[:,1]
   C = ['r' for i in range(K)]
   delay = 10
   Pic.DrawPoints(X,Y,C)
    
if __name__ == '__main__':
    Main()
    