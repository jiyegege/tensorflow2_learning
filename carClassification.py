import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
#from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

def _parse_function(example_proto):
  features = {'label':tf.FixedLenFeature([], tf.int64),
              'img_raw':tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
  img = tf.reshape(img, [120, 120, 3])
    # 在流中抛出img张量和label张量
  img = tf.cast(img, tf.float32) / 255
  label = tf.cast(parsed_features['label'], tf.int32)
  return img, label

filenames = ["CarClassify_data/train.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

test_dataset = tf.data.TFRecordDataset(['CarClassify_data/test.tfrecords'])
test_dataset = test_dataset.map(_parse_function)
# 创建单次迭代器
iterator = dataset.make_one_shot_iterator()

class_names = ['自行车', '汽 车', '摩托车', '货 车']

# 读取并显示一张图像数据
# with tf.Session() as sess:
#     for i in range(1):
#         image, label = sess.run(iterator.get_next())
#         plt.imshow(image)
#         plt.show()

#定义模型
model = keras.Sequential([
    # 第一层卷积，最大池化，dropout
    keras.layers.Conv2D(64, (5, 5), activation="relu", input_shape=(120, 120, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.15),
    # 第二层卷积，最大池化，dropout
    keras.layers.Conv2D(128, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    # 第三层卷积，最大池化，dropout
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    # 全连接层
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=4, activation='softmax')
])

# 设置模型优化策略并训练模型
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

model.fit(
    dataset.shuffle(1000).batch(100),
    epochs=10,
    verbose=1
)

# 评估模型
results = model.evaluate(test_dataset.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
predict = model.predict(test_dataset)

print(predict[0])

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

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
test_img, test_lables = test_dataset
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predict, test_lables, test_img)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
plt.tight_layout()
plt.savefig("15_pre.png")
plt.show()