import tensorflow as tf
import numpy as np

path = "MNIST_data/mnist.npz"

#加载mnist数据
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

#读取训练数据和测试数据，将样本从整数转换为浮点数

x_train, x_test = x_train / 255.0, x_test / 255.0

# #加载mnist数据
# mnist = tf.keras.datasets.mnist
#
# #读取训练数据和测试数据，将样本从整数转换为浮点数
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

#tf.keras.models.Sequential为Keras提供的层叠模型
model = tf.keras.models.Sequential([
  #输入层，输入的shape是28X28（图片大小是28X28）
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #隐藏层一，输出shape是128，激励函数是relu
  tf.keras.layers.Dense(128, activation='relu'),
  #Dropout一些神经元，能够有效地处理在训练过程中产生的过拟合问题
  tf.keras.layers.Dropout(0.2),
  #输出层，shape为10分别对应着0-9这几个手写数字，激励函数为softmax，将特征值转换为在各个分类上的概率
  tf.keras.layers.Dense(10, activation='softmax')
])

#采用adam优化器，损失函数为sparse_categorical_crossentropy，评价函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#输入数据x_train,y_train;训练次数5次
model.fit(x_train, y_train, epochs=5)
#测试模型
model.evaluate(x_test, y_test)