import cv2 as cv
from network import NeuralNetwork,Conv2D
import numpy as np

Conv = Conv2D([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
model = NeuralNetwork()
model.Load("model_cross_down.net")
cnt = 0
for i in range(10):
    img = cv.imread("picture"+str(i)+".png")
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(28,28)).reshape((28,28))
    # img = 255 - img
    result = model.Predict(Conv.Result(img).reshape(-1,1)).reshape(1,-1)
    result = np.argmax(result)
    print(result)
    if(result == i):
        cnt += 1
print("Accuracy = %f%%"%(cnt*10))
