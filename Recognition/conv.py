# Do Conv-2D on the input image
import numpy as np
import cv2 as cv

def Relu(x):
    return np.maximum(0,x)
class Conv2D:
    filter_size = 3
    filter = np.random.randn(filter_size,filter_size)
    padding = 1
    stride = 1
    
    def __init__(self,filter):
        filter = np.array(filter)
        self.filter = filter
        self.filter_size = filter.shape[0]
    
    def Conv(self,img):
        img = np.array(img)
        img = np.pad(img,((self.padding,self.padding),(self.padding,self.padding)),'constant')
        img_size = img.shape[0]
        output_size = int((img_size-self.filter_size)/self.stride)+1
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.sum(img[i:i+self.filter_size,j:j+self.filter_size]*self.filter)
        return output
    
    def Pooling_Max(self,data,size=2):
        N = data.shape[0]
        output_size = int(N/size)
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.max(data[i*size:(i+1)*size,j*size:(j+1)*size])
        return output
    
    def Pooling_Avg(self,data,size=2):
        N = data.shape[0]
        output_size = int(N/size)
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.mean(data[i*size:(i+1)*size,j*size:(j+1)*size])
        return output
    
    def Result(self,image):
        result = self.Pooling_Max(self.Conv(image))
        return result / 255.0
    
def Process(Path):
    img = cv.imread(Path)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(28,28))
    img = img.reshape((28,28))
    return 255 - img

def test(img):
    cv.imshow("img",img)
    cv.waitKey(0)
    conv1 = Conv2D([[1,0,-1],[1,0,-1],[1,0,-1]])
    conv2 = Conv2D([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    output2 = conv2.Result(img) / 255
    cv.imshow("img",cv.resize(output2,(280,280)))
    cv.waitKey(0)
    cv.waitKey(0)

def Load():
    train_data,train_label = [],[]
    Path = "./Images/"
    for i in range(1):
        for j in range(300):
            train_data.append(Process(Path+str(i)+"/"+str(j)+".png"))
            train_label.append(i)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test(train_data[np.random.randint(0,train_data.shape[0])])
    
if __name__ == "__main__":
    Load()