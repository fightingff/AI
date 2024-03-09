import tensorflow as tf
import numpy as np
import cv2 as cv

id = ['♣ 2','♣ 9','♣ 10','♣ A','♣ K','♣ Q','♦ 10','♦ A','♦ K','♦ Q','♥ 10','♥ A','♥ K','♥ Q','♠ 2','♠ 9','♠ 10','♠ A','♠ K','♠ Q']

Path = "./Images/"+str(np.random.randint(0,20))+"/"+str(np.random.randint(0,3000))+".png"
img = cv.imread(Path)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img = cv.resize(img,(28,28)).reshape((28,28))
cv.imwrite("test.png",img)

model = tf.keras.models.load_model('model.h5')
result = model.predict(img.reshape(-1,28,28,1))
result = np.argmax(result)

print(id[result])
# cv.imshow(id[result],cv.resize(img,(280,280)))
cv.waitKey(0)

