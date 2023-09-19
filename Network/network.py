# 导入TensorFlow库和MNIST数据集
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_data, train_target), (test_data, test_target) = mnist.load_data()

# 转换数据维度
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

# 归一化
train_data = train_data / 255.0
test_data = test_data / 255.0

# 独热编码
train_target = tf.keras.utils.to_categorical(train_target, num_classes=10)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=10)

# 定义卷积神经网络模型
model = tf.keras.Sequential()
# 第一个卷积层
model.add(tf.keras.layers.Convolution2D(input_shape=(28, 28, 1), filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
# 第一个池化层
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
# 第二个卷积层
model.add(tf.keras.layers.Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
# 第二个池化层
model.add(tf.keras.layers.MaxPooling2D(2, 2, 'same'))
# 扁平化
model.add(tf.keras.layers.Flatten())
# 第一个全连接层
model.add(tf.keras.layers.Dense(1024, activation='relu'))
# 第二个全连接层（输出层）
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_target, batch_size=64, epochs=10, validation_data=(test_data, test_target))

# 测试模型
model.evaluate(test_data, test_target)

# 保存模型
model.save('model.h5')



