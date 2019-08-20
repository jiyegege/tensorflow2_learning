#导入相关包
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#导入时装MNIST数据集，作为MNIST手写时装数据集的扩展
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#print(train_images.shape)
#print(len(train_images))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()
# #显示图像
# plt.imshow(train_images[0])
# #颜色比例彩条
# plt.colorbar()
# #不显示网格线
# plt.grid(False)
# #保存图像
# plt.savefig('some_one.png')
# #显示图像
# plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

# #创建图
# plt.figure(figsize=(10,10))
# for i in range(25):
#     #创建子图，5行5列，第i+1个
#     plt.subplot(5,5,i+1)
#     #X轴坐标刻度设置
#     plt.xticks([])
#     #Y轴坐标刻度设置
#     plt.yticks([])
#     plt.grid(False)
#     #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间，这里使用的是二值图
#     plt.imshow(train_images[i])
#     plt.colorbar()
#     #设置X轴标签
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
model = keras.Sequential([
    # 输入层
    keras.layers.Flatten(input_shape=(28, 28)),
    # 隐藏层一，全连接
    keras.layers.Dense(128, activation='relu'),
    # 输出层，全连接
    keras.layers.Dense(10, activation='softmax')
])
#采用Adam优化器，loss计算方法为sparse_categorical_crossentropy，评价函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#训练总轮数为5轮
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)
#print(predictions[0])
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(range(10), class_names, rotation=90)
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.tight_layout()
plt.savefig("15_pre.png")
plt.show()